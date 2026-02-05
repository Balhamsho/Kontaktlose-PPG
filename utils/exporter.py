import json 
import pandas as pd 
import os 
from datetime import datetime 

def export_results(results):
    os.makedirs("data/results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") 

    json_path = f"data/results/result_{ts}.json" 
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4) 

    rows = [] 
    for method, data in results["methods"].items(): 
        for t, s in zip(data["timestamps"], data["signal"]): 
            rows.append({
                "method": method,
                "time": t, 
                "signal": s, 
                "bpm": data["bpm"], 
                "video": data["video_path"] 
            })

    df = pd.DataFrame(rows) 
    csv_path = f"data/results/result_{ts}.csv" 
    df.to_csv(csv_path, index=False) 

    print(f"Saved JSON → {json_path}") 
    print(f"Saved CSV  → {csv_path}") 

    return json_path, csv_path 
