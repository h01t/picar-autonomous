import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_joystick/flutter_joystick.dart';
import 'package:http/http.dart' as http;

const String PI_IP = "192.168.4.1";

void main() => runApp(const PiCarApp());

class PiCarApp extends StatelessWidget {
  const PiCarApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: const ModeSelect(),
    );
  }
}

class ModeSelect extends StatelessWidget {
  const ModeSelect({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("PiCar Controller")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              child: const Text("Manual / Collect Drive"),
              onPressed: () async {
                // Set orientation BEFORE navigation
                await SystemChrome.setPreferredOrientations([
                  DeviceOrientation.landscapeLeft,
                  DeviceOrientation.landscapeRight
                ]);

                if (context.mounted) {
                  Navigator.push(context,
                      MaterialPageRoute(builder: (_) => const ManualDrive()));
                }
              },
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              child: const Text("Autonomous"),
              onPressed: () {
                Navigator.push(context,
                    MaterialPageRoute(builder: (_) => const Autonomous()));
              },
            ),
          ],
        ),
      ),
    );
  }
}

class ManualDrive extends StatefulWidget {
  const ManualDrive({super.key});
  @override
  State<ManualDrive> createState() => _ManualDriveState();
}

class _ManualDriveState extends State<ManualDrive> {
  late Socket control;
  late Socket telemetry;
  Timer? _timer;
  StreamSubscription? _streamSubscription;
  http.Client? _httpClient;

  double steering = 0.0;
  double throttle = 0.0;

  String fps = "0";
  String mode = "manual";
  String connectionStatus = "Connecting...";
  bool obstacleDetected = false;
  Map<String, double> sensorDistances = {
    'front': 999.0,
    'left': 999.0,
    'right': 999.0,
    'rear': 999.0
  };

  // For MJPEG streaming
  List<int> _imageBytes = [];
  List<int> _currentFrame = [];
  bool _isStreamActive = false;

  @override
  void initState() {
    super.initState();

    print("[DEBUG] ManualDrive initState called");

    // 1. Connect Control Socket
    Socket.connect(PI_IP, 5005, timeout: const Duration(seconds: 5)).then((s) {
      control = s;
      setState(() => connectionStatus = "Connected");
      print("[DEBUG] Control socket connected");
    }).catchError((e) {
      setState(() => connectionStatus = "Control Failed: $e");
      print("[ERROR] Control socket failed: $e");
    });

    // 2. Connect Telemetry Socket
    Socket.connect(PI_IP, 5006, timeout: const Duration(seconds: 5)).then((s) {
      telemetry = s;
      print("[DEBUG] Telemetry socket connected");

      String buffer = '';

      telemetry.listen((data) {
        try {
          // Append to buffer
          buffer += utf8.decode(data);

          // Process complete JSON objects (ending with newline)
          while (buffer.contains('\n')) {
            int newlineIndex = buffer.indexOf('\n');
            String jsonStr = buffer.substring(0, newlineIndex).trim();
            buffer = buffer.substring(newlineIndex + 1);

            if (jsonStr.isEmpty) continue;

            final msg = jsonDecode(jsonStr);
            if (mounted) {
              setState(() {
                fps = msg["fps"].toString();
                steering = msg["steering"];
                throttle = msg["throttle"];
                mode = msg["mode"];
                obstacleDetected = msg["obstacle"] ?? false;

                // Update sensor distances
                if (msg["sensors"] != null) {
                  sensorDistances['front'] =
                      (msg["sensors"]["front"] ?? 999.0).toDouble();
                  sensorDistances['left'] =
                      (msg["sensors"]["left"] ?? 999.0).toDouble();
                  sensorDistances['right'] =
                      (msg["sensors"]["right"] ?? 999.0).toDouble();
                  sensorDistances['rear'] =
                      (msg["sensors"]["rear"] ?? 999.0).toDouble();
                }
              });
            }
          }
        } catch (e) {
          print("[ERROR] Telemetry parse error: $e");
        }
      });
    }).catchError((e) {
      print("[ERROR] Telemetry socket failed: $e");
    });

    // 3. Start Heartbeat Loop
    _timer = Timer.periodic(const Duration(milliseconds: 50), (_) {
      try {
        control.write(jsonEncode(
                {"steering": steering, "throttle": throttle, "mode": mode}) +
            "\n");
      } catch (e) {
        // Socket might not be ready yet
      }
    });

    // 4. Start MJPEG stream with delay
    Future.delayed(const Duration(milliseconds: 500), () {
      if (mounted) {
        print("[DEBUG] Starting MJPEG stream...");
        _startMjpegStream();
      }
    });
  }

  void _startMjpegStream() {
    if (_isStreamActive) {
      print("[DEBUG] Stream already active, skipping");
      return;
    }

    print("[DEBUG] Initializing MJPEG stream...");
    _isStreamActive = true;
    _imageBytes = [];
    _currentFrame = [];

    _httpClient = http.Client();
    final request = http.Request('GET', Uri.parse('http://$PI_IP:8080/stream'));

    print("[DEBUG] Sending HTTP request to http://$PI_IP:8080/stream");

    _httpClient!.send(request).then((response) {
      print("[DEBUG] Got HTTP response, status: ${response.statusCode}");

      _streamSubscription = response.stream.listen(
        (chunk) {
          if (!mounted) return;

          _imageBytes.addAll(chunk);

          // Look for JPEG boundaries
          final startMarker = [0xFF, 0xD8]; // JPEG start
          final endMarker = [0xFF, 0xD9]; // JPEG end

          int startIndex = -1;
          int endIndex = -1;

          for (int i = 0; i < _imageBytes.length - 1; i++) {
            if (_imageBytes[i] == startMarker[0] &&
                _imageBytes[i + 1] == startMarker[1]) {
              startIndex = i;
            }
            if (_imageBytes[i] == endMarker[0] &&
                _imageBytes[i + 1] == endMarker[1]) {
              endIndex = i + 1;
              break;
            }
          }

          if (startIndex != -1 && endIndex != -1 && endIndex > startIndex) {
            final imageData = _imageBytes.sublist(startIndex, endIndex + 1);
            if (mounted) {
              setState(() {
                _currentFrame = imageData;
              });
              if (_currentFrame.length > 0 && _currentFrame.length % 10 == 0) {
                print(
                    "[DEBUG] Received frame, size: ${_currentFrame.length} bytes");
              }
            }
            _imageBytes = _imageBytes.sublist(endIndex + 1);
          }
        },
        onError: (e) {
          print("[ERROR] Stream error: $e");
          _isStreamActive = false;
          // Attempt to reconnect after 2 seconds
          if (mounted) {
            Future.delayed(const Duration(seconds: 2), () {
              if (mounted) {
                print("[DEBUG] Attempting to reconnect stream...");
                _startMjpegStream();
              }
            });
          }
        },
        onDone: () {
          print("[DEBUG] Stream ended");
          _isStreamActive = false;
        },
        cancelOnError: false,
      );
    }).catchError((e) {
      print("[ERROR] Connection error: $e");
      _isStreamActive = false;
      // Attempt to reconnect after 2 seconds
      if (mounted) {
        Future.delayed(const Duration(seconds: 2), () {
          if (mounted) {
            print("[DEBUG] Attempting to reconnect after error...");
            _startMjpegStream();
          }
        });
      }
    });
  }

  @override
  void dispose() {
    print("[DEBUG] ManualDrive disposing...");
    _timer?.cancel();
    _streamSubscription?.cancel();
    _httpClient?.close();
    try {
      control.destroy();
      telemetry.destroy();
    } catch (e) {
      print("[ERROR] Error closing sockets: $e");
    }
    // Reset orientation when leaving
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
    super.dispose();
  }

  Color _getSensorColor(double distance) {
    if (distance > 30) return Colors.green;
    if (distance > 15) return Colors.yellow;
    return Colors.red;
  }

  Widget _buildSensorOverlay() {
    print(
        "[DEBUG] Building sensor overlay - Front:${sensorDistances['front']} Left:${sensorDistances['left']} Right:${sensorDistances['right']} Rear:${sensorDistances['rear']}");

    return Container(
      width: 140,
      height: 140,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white, width: 2),
      ),
      child: Stack(
        children: [
          // Title
          const Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Text(
              'SENSORS',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
          ),

          // Car in center
          Center(
            child: Container(
              width: 40,
              height: 50,
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.3),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue, width: 2),
              ),
              child: const Icon(
                Icons.directions_car,
                color: Colors.white,
                size: 30,
              ),
            ),
          ),

          // Front sensor
          Positioned(
            top: 20,
            left: 0,
            right: 0,
            child: Column(
              children: [
                Container(
                  width: 30,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _getSensorColor(sensorDistances['front']!),
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${sensorDistances['front']!.toInt()}cm',
                  style: TextStyle(
                    fontSize: 9,
                    color: _getSensorColor(sensorDistances['front']!),
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),

          // Left sensor
          Positioned(
            left: 10,
            top: 0,
            bottom: 0,
            child: Center(
              child: Row(
                children: [
                  Text(
                    '${sensorDistances['left']!.toInt()}',
                    style: TextStyle(
                      fontSize: 9,
                      color: _getSensorColor(sensorDistances['left']!),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(width: 2),
                  Container(
                    width: 8,
                    height: 30,
                    decoration: BoxDecoration(
                      color: _getSensorColor(sensorDistances['left']!),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Right sensor
          Positioned(
            right: 10,
            top: 0,
            bottom: 0,
            child: Center(
              child: Row(
                children: [
                  Container(
                    width: 8,
                    height: 30,
                    decoration: BoxDecoration(
                      color: _getSensorColor(sensorDistances['right']!),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  const SizedBox(width: 2),
                  Text(
                    '${sensorDistances['right']!.toInt()}',
                    style: TextStyle(
                      fontSize: 9,
                      color: _getSensorColor(sensorDistances['right']!),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Rear sensor
          Positioned(
            bottom: 20,
            left: 0,
            right: 0,
            child: Column(
              children: [
                Text(
                  '${sensorDistances['rear']!.toInt()}cm',
                  style: TextStyle(
                    fontSize: 9,
                    color: _getSensorColor(sensorDistances['rear']!),
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 2),
                Container(
                  width: 30,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _getSensorColor(sensorDistances['rear']!),
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Force rebuild when orientation changes
    final size = MediaQuery.of(context).size;
    final isLandscape = size.width > size.height;

    print(
        "[DEBUG] Building ManualDrive - Size: ${size.width}x${size.height}, Landscape: $isLandscape, HasFrame: ${_currentFrame.isNotEmpty}");

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // CAMERA STREAM BACKGROUND
          Positioned.fill(
            child: _currentFrame.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const CircularProgressIndicator(color: Colors.white),
                        const SizedBox(height: 16),
                        Text(
                          "Waiting for camera stream...\n$connectionStatus",
                          textAlign: TextAlign.center,
                          style: const TextStyle(color: Colors.white),
                        ),
                      ],
                    ),
                  )
                : Image.memory(
                    Uint8List.fromList(_currentFrame),
                    fit: BoxFit.cover,
                    gaplessPlayback: true,
                    errorBuilder: (context, error, stackTrace) {
                      print("[ERROR] Image display error: $error");
                      return Center(
                        child: Text(
                          "Camera Error: $error",
                          style: const TextStyle(color: Colors.red),
                        ),
                      );
                    },
                  ),
          ),

          // TELEMETRY HUD
          Positioned(
            top: 12,
            left: 12,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Status: $connectionStatus"),
                  Text("FPS: $fps"),
                  Text("Steering: ${steering.toStringAsFixed(2)}"),
                  Text("Throttle: ${throttle.toStringAsFixed(2)}"),
                  Text("Mode: $mode"),
                  if (obstacleDetected)
                    const Text(
                      "⚠️ OBSTACLE!",
                      style: TextStyle(
                        color: Colors.red,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                ],
              ),
            ),
          ),

          // STEERING JOYSTICK
          Positioned(
            left: 20,
            bottom: 20,
            child: Container(
              decoration: BoxDecoration(
                color: Colors.black26,
                shape: BoxShape.circle,
              ),
              child: Joystick(
                mode: JoystickMode.horizontal,
                listener: (details) {
                  steering = details.x;
                },
              ),
            ),
          ),

          // THROTTLE JOYSTICK
          Positioned(
            right: 20,
            bottom: 20,
            child: Container(
              decoration: BoxDecoration(
                color: Colors.black26,
                shape: BoxShape.circle,
              ),
              child: Joystick(
                mode: JoystickMode.vertical,
                listener: (details) {
                  throttle = (-details.y).clamp(0.0, 1.0);
                },
              ),
            ),
          ),

          // BACK BUTTON
          Positioned(
            top: 20,
            right: 20,
            child: FloatingActionButton(
              mini: true,
              backgroundColor: Colors.red,
              child: const Icon(Icons.close),
              onPressed: () => Navigator.pop(context),
            ),
          ),

          // SENSOR OVERLAY (only in autonomous mode)
          if (mode == "autonomous")
            Positioned(
              bottom: 120,
              right: 20,
              child: _buildSensorOverlay(),
            ),
        ],
      ),
    );
  }
}

class Autonomous extends StatefulWidget {
  const Autonomous({super.key});
  @override
  State<Autonomous> createState() => _AutonomousState();
}

class _AutonomousState extends State<Autonomous> {
  bool enabled = false;
  Socket? control; // Changed to nullable
  Socket? telemetry;
  Timer? _timer;
  String connectionStatus = "Connecting...";
  bool _isDisposed = false; // Track if widget is disposed
  int _reconnectAttempts = 0; // Track reconnection attempts for backoff
  bool _controlConnected = false; // Track if control socket is connected

  // Add telemetry for sensor display
  double steering = 0.0;
  double throttle = 0.0;
  String mode = "manual";
  Map<String, double> sensorDistances = {
    'front': 999.0,
    'left': 999.0,
    'right': 999.0,
    'rear': 999.0
  };

  @override
  void initState() {
    super.initState();

    // Connect control socket
    Socket.connect(PI_IP, 5005, timeout: const Duration(seconds: 5)).then((s) {
      control = s;
      _controlConnected = true;
      setState(() => connectionStatus = "Connected");

      // Send periodic heartbeat
      _timer = Timer.periodic(const Duration(milliseconds: 50), (_) {
        // Only write if control socket is connected
        if (!_controlConnected || control == null) return;

        try {
          if (enabled) {
            control!.write(jsonEncode(
                    {"steering": 0.0, "throttle": 0.0, "mode": "autonomous"}) +
                "\n");
          } else {
            control!.write(jsonEncode(
                    {"steering": 0.0, "throttle": 0.0, "mode": "manual"}) +
                "\n");
          }
        } catch (e) {
          // Socket write failed - mark as disconnected
          print("[ERROR] Control socket write failed: $e");
          _controlConnected = false;
        }
      });
    }).catchError((e) {
      setState(() => connectionStatus = "Failed: $e");
    });

    // Connect telemetry socket to get sensor data
    _connectTelemetry();
  }

  void _connectTelemetry() {
    if (_isDisposed) return;

    // Close existing connection if any
    try {
      telemetry?.destroy();
    } catch (e) {}

    print(
        "[DEBUG] Attempting to connect telemetry socket (attempt ${_reconnectAttempts + 1})...");

    Socket.connect(PI_IP, 5006, timeout: const Duration(seconds: 5)).then((s) {
      if (_isDisposed) {
        s.destroy();
        return;
      }

      telemetry = s;
      _reconnectAttempts = 0; // Reset on successful connection
      print("[DEBUG] Autonomous telemetry socket connected");
      String buffer = '';

      s.listen(
        (data) {
          try {
            buffer += utf8.decode(data);

            while (buffer.contains('\n')) {
              int newlineIndex = buffer.indexOf('\n');
              String jsonStr = buffer.substring(0, newlineIndex).trim();
              buffer = buffer.substring(newlineIndex + 1);

              if (jsonStr.isEmpty) continue;

              final msg = jsonDecode(jsonStr);
              print(
                  "[DEBUG] Telemetry received at ${DateTime.now().millisecondsSinceEpoch % 100000}: mode=${msg["mode"]}");
              if (mounted) {
                setState(() {
                  mode = msg["mode"] ?? "manual";
                  steering = msg["steering"] ?? 0.0;
                  throttle = msg["throttle"] ?? 0.0;

                  if (msg["sensors"] != null) {
                    sensorDistances['front'] =
                        (msg["sensors"]["front"] ?? 999.0).toDouble();
                    sensorDistances['left'] =
                        (msg["sensors"]["left"] ?? 999.0).toDouble();
                    sensorDistances['right'] =
                        (msg["sensors"]["right"] ?? 999.0).toDouble();
                    sensorDistances['rear'] =
                        (msg["sensors"]["rear"] ?? 999.0).toDouble();
                  }
                });
              }
            }
          } catch (e) {
            print("[ERROR] Telemetry parse: $e");
          }
        },
        onError: (e) {
          print("[ERROR] Autonomous telemetry stream error: $e");
          _scheduleReconnect();
        },
        onDone: () {
          print("[DEBUG] Autonomous telemetry stream closed by server");
          _scheduleReconnect();
        },
        cancelOnError: false,
      );
    }).catchError((e) {
      print("[ERROR] Autonomous telemetry socket failed: $e");
      _scheduleReconnect();
    });
  }

  void _scheduleReconnect() {
    if (_isDisposed || !mounted) return;

    _reconnectAttempts++;
    // Exponential backoff: 1s, 2s, 4s, 8s, max 10s
    int delaySeconds = (_reconnectAttempts < 4)
        ? (1 << (_reconnectAttempts - 1)) // 1, 2, 4, 8
        : 10;

    print(
        "[DEBUG] Scheduling reconnect in ${delaySeconds}s (attempt $_reconnectAttempts)");

    Future.delayed(Duration(seconds: delaySeconds), () {
      if (!_isDisposed && mounted) {
        _connectTelemetry();
      }
    });
  }

  @override
  void dispose() {
    _isDisposed = true; // Prevent reconnection attempts
    _timer?.cancel();

    // Stop the car before disconnecting
    if (_controlConnected && control != null) {
      try {
        control!.write(
            jsonEncode({"steering": 0.0, "throttle": 0.0, "mode": "manual"}) +
                "\n");
        control!.destroy();
      } catch (e) {
        print("[ERROR] Error closing control socket: $e");
      }
    }

    // Close telemetry socket too!
    try {
      telemetry?.destroy();
      print("[DEBUG] Autonomous telemetry socket closed");
    } catch (e) {}

    super.dispose();
  }

  Color _getSensorColor(double distance) {
    if (distance > 30) return Colors.green;
    if (distance > 15) return Colors.yellow;
    return Colors.red;
  }

  Widget _buildSensorOverlay() {
    print("[DEBUG] Building sensor overlay - Mode: $mode, Enabled: $enabled");
    print(
        "[DEBUG] Distances - F:${sensorDistances['front']} L:${sensorDistances['left']} R:${sensorDistances['right']} Re:${sensorDistances['rear']}");

    // Only show when enabled
    if (!enabled) return const SizedBox.shrink();

    return Container(
      width: 140,
      height: 140,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white, width: 2),
      ),
      child: Stack(
        children: [
          // Title
          const Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Text(
              'SENSORS',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
          ),

          // Car in center
          Center(
            child: Container(
              width: 40,
              height: 50,
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.3),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue, width: 2),
              ),
              child: const Icon(
                Icons.directions_car,
                color: Colors.white,
                size: 30,
              ),
            ),
          ),

          // Front sensor
          Positioned(
            top: 20,
            left: 0,
            right: 0,
            child: Column(
              children: [
                Container(
                  width: 30,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _getSensorColor(sensorDistances['front']!),
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${sensorDistances['front']!.toInt()}cm',
                  style: TextStyle(
                    fontSize: 9,
                    color: _getSensorColor(sensorDistances['front']!),
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),

          // Left sensor
          Positioned(
            left: 10,
            top: 0,
            bottom: 0,
            child: Center(
              child: Row(
                children: [
                  Text(
                    '${sensorDistances['left']!.toInt()}',
                    style: TextStyle(
                      fontSize: 9,
                      color: _getSensorColor(sensorDistances['left']!),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(width: 2),
                  Container(
                    width: 8,
                    height: 30,
                    decoration: BoxDecoration(
                      color: _getSensorColor(sensorDistances['left']!),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Right sensor
          Positioned(
            right: 10,
            top: 0,
            bottom: 0,
            child: Center(
              child: Row(
                children: [
                  Container(
                    width: 8,
                    height: 30,
                    decoration: BoxDecoration(
                      color: _getSensorColor(sensorDistances['right']!),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  const SizedBox(width: 2),
                  Text(
                    '${sensorDistances['right']!.toInt()}',
                    style: TextStyle(
                      fontSize: 9,
                      color: _getSensorColor(sensorDistances['right']!),
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Rear sensor
          Positioned(
            bottom: 20,
            left: 0,
            right: 0,
            child: Column(
              children: [
                Text(
                  '${sensorDistances['rear']!.toInt()}cm',
                  style: TextStyle(
                    fontSize: 9,
                    color: _getSensorColor(sensorDistances['rear']!),
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 2),
                Container(
                  width: 30,
                  height: 8,
                  decoration: BoxDecoration(
                    color: _getSensorColor(sensorDistances['rear']!),
                    borderRadius: BorderRadius.circular(4),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Autonomous Mode")),
      body: Stack(
        children: [
          Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(16),
                child: Text(
                  "Status: $connectionStatus",
                  style: TextStyle(
                    color: connectionStatus.contains("Connected")
                        ? Colors.green
                        : Colors.orange,
                  ),
                ),
              ),
              SwitchListTile(
                title: const Text("Enable Autonomous Driving"),
                subtitle: Text(
                    "Mode: $mode | Steering: ${steering.toStringAsFixed(2)} | Throttle: ${throttle.toStringAsFixed(2)}"),
                value: enabled,
                onChanged: (v) {
                  setState(() {
                    enabled = v;
                    if (v) {
                      // Reset sensor data when re-enabling to make it obvious if updates stop
                      sensorDistances = {
                        'front': 999.0,
                        'left': 999.0,
                        'right': 999.0,
                        'rear': 999.0
                      };
                    }
                  });
                  print(
                      "[DEBUG] Autonomous ${v ? 'ENABLED' : 'DISABLED'} at ${DateTime.now()}");
                },
              ),
              if (enabled)
                const Padding(
                  padding: EdgeInsets.all(16),
                  child: Card(
                    color: Colors.green,
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Row(
                        children: [
                          Icon(Icons.check_circle, color: Colors.white),
                          SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              "Autonomous mode active!\nModel is steering, server provides throttle.",
                              style: TextStyle(color: Colors.white),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),

          // Sensor overlay
          if (enabled)
            Positioned(
              bottom: 20,
              right: 20,
              child: _buildSensorOverlay(),
            ),
        ],
      ),
    );
  }
}
