import os
import json
import pandas as pd
import glob

# Modificare i percorsi secondo necessità
CARTELLA_PADRE = r"./Consegna/SIMULATOR/lecture_example_from_training/test_set/videos"

IMG_WIDTH = 1920
IMG_HEIGHT = 1080


def process_single_video_folder(folder_path):
    
    gt_path = os.path.join(folder_path, 'gt', 'gt.txt')
    roi_path = os.path.join(folder_path, 'roi.json')
    gameinfo_path = os.path.join(folder_path, 'gameinfo.ini')
    output_path = os.path.join(folder_path, 'gt', 'behavior_gt.txt')

    if not (os.path.exists(gt_path) and os.path.exists(roi_path) and os.path.exists(gameinfo_path)):
        print(f"File mancanti in: {folder_path}")
        return

    # Rimozione id ball dal gameinfo.ini
    ball_ids = []
    try:
        with open(gameinfo_path, 'r') as f:
            for line in f:
                if 'trackletID_' in line and 'ball' in line.lower():
                    left_part = line.split('=')[0].strip()
                    try:
                        tid = int(left_part.split('_')[1])
                        ball_ids.append(tid)
                    except (IndexError, ValueError):
                        continue
    except Exception as e:
        print(f"Errore lettura gameinfo in {folder_path}: {e}")
        return

    processed_rois = {}
    try:
        with open(roi_path, 'r') as f:
            rois_data = json.load(f)
            
        for key, val in rois_data.items():
            roi_idx = int(key.lower().replace('roi', ''))
            
            processed_rois[roi_idx] = {
                'x1': val['x'] * IMG_WIDTH,
                'y1': val['y'] * IMG_HEIGHT,
                'x2': (val['x'] + val['width']) * IMG_WIDTH,
                'y2': (val['y'] + val['height']) * IMG_HEIGHT
            }
    except Exception as e:
        print(f"Errore lettura ROI in {folder_path}: {e}")
        return

    # Lettura del file gt.txt
    try:
        df = pd.read_csv(gt_path, header=None)
        # Colonne: frame, id, x, y, w, h, conf, class, vis
        df.rename(columns={0: 'frame', 1: 'id', 2: 'x', 3: 'y', 4: 'w', 5: 'h'}, inplace=True)
    except Exception as e:
        print(f"Errore lettura GT in {folder_path}: {e}")
        return

    if ball_ids:
        df = df[~df['id'].isin(ball_ids)]

    df['center_x'] = df['x'] + (df['w'] / 2)
    df['center_y'] = df['y'] + df['h']

    results = []
    frames = sorted(df['frame'].unique())

    for frame in frames:
        current_frame_data = df[df['frame'] == frame]
        
        for roi_id, coords in processed_rois.items():
            count = current_frame_data[
                (current_frame_data['center_x'] >= coords['x1']) &
                (current_frame_data['center_x'] <= coords['x2']) &
                (current_frame_data['center_y'] >= coords['y1']) &
                (current_frame_data['center_y'] <= coords['y2'])
            ].shape[0]
            
            results.append([frame, roi_id, count])

    results.sort(key=lambda x: (x[0], x[1]))
    
    try:
        with open(output_path, 'w') as f_out:
            for row in results:
                f_out.write(f"{row[0]},{row[1]},{row[2]}\n")
        print(f"Finito: {os.path.basename(folder_path)}")
    except Exception as e:
        print(f"Errore scrittura output in {folder_path}: {e}")

# --- MAIN ---
if __name__ == "__main__":
    if not os.path.exists(CARTELLA_PADRE):
        print(f"La cartella specificata non esiste:\n{CARTELLA_PADRE}")
    else:
        print(f"Analisi cartella: {CARTELLA_PADRE}")
        # Cerca tutte le sottocartelle
        subfolders = [f.path for f in os.scandir(CARTELLA_PADRE) if f.is_dir()]
        
        for folder in subfolders:
            process_single_video_folder(folder)
            