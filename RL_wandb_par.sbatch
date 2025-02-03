#!/bin/bash
#SBATCH --job-name=RL_cyc_wCorr_wandb
#SBATCH --error=RL_cyc_wCorr_wandb.err
#SBATCH --output=RL_cyc_wCorr_wandb.out
#SBATCH --nodes=1
#SBATCH --partition=work
#SBATCH --ntasks=10  # Number of agents
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=jkratz@andrew.cmu.edu
#SBATCH --nodelist=c04

SWEEP_INFO_FILE="wandb_sweep_info.txt"

# Initialize the sweep if not already done
if [ ! -f "$SWEEP_INFO_FILE" ]; then
  echo "Initializing W&B sweep..."
  SWEEP_ID=$(python3 - <<END
import wandb
wandb.login()
sweep_config = {
    'method': 'grid',
    'parameters': {
        'batch_size': {'values': [128, 256, 512]},
        'delay_embed_len': {'values': [10, 20, 30]}
    }
}
sweep_id = wandb.sweep(sweep_config, project="antibioticRL-test")
print(sweep_id)
END
)
  echo "$SWEEP_ID" > "$SWEEP_INFO_FILE"
else
  SWEEP_ID=$(cat "$SWEEP_INFO_FILE")
fi

# Wait for the sweep ID file to be created
while [ ! -f "$SWEEP_INFO_FILE" ]; do
  sleep 1
done

# Start each W&B agent in parallel
for i in $(seq 1 $SLURM_NTASKS); do
  echo "Starting W&B agent $i for sweep ID: $SWEEP_ID"
  python3 -m wandb agent "antibioticRL-test/$SWEEP_ID" &
done

# Wait for all background processes to complete
wait