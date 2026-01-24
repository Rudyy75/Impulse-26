
import os
import subprocess
import sys

def run_step(module_name, step_name):
    print(f"\n🚀 Running {step_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", module_name])
        print(f"✅ {step_name} Complete.")
    except subprocess.CalledProcessError:
        print(f"❌ {step_name} Failed!")
        sys.exit(1)

def main():
    print("================================================================")
    print(" 🏆 PHASE 4: FINAL EVALUATION (F1 Score & Visualization)")
    print("================================================================")
    print("This script executes the Organizer's Evaluation Strategy:")
    print("1. Build Vector Database (Phase 3 Indexing)")
    print("2. Linear Probe (F1 Score > 0.6 check)")
    print("3. Latent Space Visualization (t-SNE Plot)\n")
    
    # 1. Build Index (Dependency for Linear Probe)
    # run_step("src.index_db", "Vector Database Indexing")
    
    # 2. Linear Probe (F1 Score)
    run_step("src.linear_probe", "Linear Probe Evaluation (F1 Score)")
    
    # 3. Visualization
    run_step("src.visualize", "t-SNE Visualization")
    
    print("\n----------------------------------------------------------------")
    print("🎉 Evaluation Complete!")
    print("1. Check F1 Score above.")
    print("2. Check 'tsne_clusters.png' for visual clusters.")
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    main()
