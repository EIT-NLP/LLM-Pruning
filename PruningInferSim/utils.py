import os
import pandas as pd

from typing import Dict, Tuple, List, Optional

def record_results(res_dict: Dict[Tuple[int, int], List[float]], output_file: str, prefix: str, dir: Optional[str]="./metric_results"):
    df = pd.DataFrame(columns=[
        "Batch Size",
        "Seq Len",
        "Sparsity",
        "Throughput (token/s)",
        "Max Throughput (token/s)",
        "Min Throughput (token/s)",
    ])

    for (batch_size, seq_len, sparsity, _), res in res_dict.items():
        df.loc[len(df)] = [batch_size, seq_len, sparsity, res[0], res[2], res[1]]
    
    res = df.to_string()

    print(prefix)
    print(res)

    os.makedirs(dir, exist_ok=True)
    output_file = os.path.join(dir, output_file)
    # record to tmp txt, if exist then add to the end, if not exist then create
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(prefix + '\n')
            f.write(res + '\n')
    else:
        with open(output_file, "a") as f:
            f.write(prefix + '\n')
            f.write(res + '\n')