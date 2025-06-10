#!/bin/bash
mkdir -p app/model

gdown --id 1y3z79VfkMRs-frZIoy4R-dLLYhiCF3iS -O app/model/modelo_bert.pth

gdown --id 1cy8BNQcBM_thnPnhT7-7lljQHbMkil0s -O app/model/modelo_gpt2.pth
