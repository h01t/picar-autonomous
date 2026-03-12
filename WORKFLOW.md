Session 1 (20 min): Corridor loops
- Drive continuously in figure-8 patterns
- Vary speed between 0.4-0.7
- Don't stop unless emergency

Session 2 (20 min): Obstacle navigation  
- Set up boxes/cones at varying distances
- Navigate around them at moderate speed
- Include tight turns

Session 3 (15 min): Edge cases
- Drive close to walls (but not touching)
- Recovery from near-collisions
- Sharper turns at intersections


🔧 Workflow
1. Data Collection (on Pi)
python3 collect_data_v2.py  # 20-30 min session

2. Transfer to PC
scp -r pi@192.168.4.1:~/dataset ./

3. Clean Data (on PC)
python3 clean_dataset.py

4. Train (on PC with GPU)
python3 train_v2.py

5. Export for Pi (on PC)
python3 export.py  # Uses model_best.pth

6. Deploy (to Pi)
scp model.pt pi@192.168.4.1:~/
