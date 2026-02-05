import os 
import json  
import csv
import matplotlib.pyplot as plt 


def save_results(results, signals, timestamps, fps, video_path, base_name):
    os.makedirs("data/results", exist_ok=True) 

    # ---------- JSON ----------
    json_path = f"data/results/{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    # ---------- CSV ----------
    csv_path = f"data/results/{base_name}.csv" 
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "green"])
        for i in range(len(timestamps)):
            writer.writerow([ 
                timestamps[i],
                signals["green"][i]
            ])

    # ---------- Plot ----------
    plt.figure() 
    plt.plot(timestamps, signals["green"], color="green") 
    plt.title("GREEN Signal") 
    plt.xlabel("Time (s)") 
    plt.ylabel("Amplitude") 
    plt.grid() 
    plt.savefig(f"data/results/{base_name}_green.png") 
    plt.close() 

    print("✅ Results saved:") 
    print(json_path) 
    print(csv_path) 
