How to create a dataset:

1. Run the create_game_csv.py script
   
   Execute the script in the Kaggle notebook with the 
   meta-kaggle dataset. This will generate the game.csv file. Download this 
   file to the imitation_learning/dataset folder.

2. Update the submissions CSV
   
   ```bash
   python update_submissions_csv.py
   ```
   
   This will generate the submissions.csv file.

3. Download episodes
   
   Run the command below, replacing xxx with the desired submission ID:
   ```bash
   python get_episodes.py --submission_id xxx
   ```
   
   This will create a folder dataset/episodes and download the 
   JSON replays from Kaggle.

4. Convert episodes for Unit-Unet

   ```bash
   python convert_episodes.py --submission_id xxx --num_workers 4
   ```
   
   This will create a folder dataset/agent_episodes and generate 
   npz files needed for Unit-Unet training.

5. Convert episodes for SAP-Unet
   
   ```bash
   python convert_episodes.py --submission_id xxx --num_workers 4 --sap
   ```
   
   This will create a folder dataset/agent_episodes_sap and 
   generate npz files needed for SAP-Unet training.
