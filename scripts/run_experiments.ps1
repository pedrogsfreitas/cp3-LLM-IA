# Roda todo o pipeline de experimentos na GPU (PowerShell, com o .venv ativado).
#   .\scripts\run_experiments.ps1
# Para rodar só uma parte, copie as linhas desejadas.
$ErrorActionPreference = "Stop"

# 1) Bancos e dataset (rápido, determinístico)
python data/build_databases.py
python data/generate_dataset.py

# 2) Linha de base: modelo sem treino
python src/evaluate.py --name base

# 3) Fine-tuning com diferentes parâmetros + avaliação de cada um
$exps = @("exp1_baseline", "exp2_lr_baixo", "exp3_rank8", "exp4_qlora", "exp5_3epocas")
foreach ($exp in $exps) {
    python src/train.py --config "configs/$exp.yaml"
    $extra = @()
    if ($exp -eq "exp4_qlora") { $extra = @("--load_in_4bit") }
    python src/evaluate.py --name $exp --adapter "outputs/$exp/adapter" @extra
}

# 4) Parâmetros de geração (no modelo baseline treinado)
python src/evaluate.py --name exp1_temp0.3 --adapter outputs/exp1_baseline/adapter --temperature 0.3 --top_p 0.9
python src/evaluate.py --name exp1_temp0.7 --adapter outputs/exp1_baseline/adapter --temperature 0.7 --top_p 0.9
python src/evaluate.py --name exp1_temp1.0 --adapter outputs/exp1_baseline/adapter --temperature 1.0 --top_p 0.95

# 5) Tabela comparativa (results/resumo.md)
python src/compare_results.py
