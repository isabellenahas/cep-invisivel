from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
ROOT=Path(__file__).resolve().parent
f=pd.read_csv(ROOT/'data/dataset_ml.csv',dtype={'CD_SETOR':str})
p=pd.read_csv(ROOT/'data/predicoes_teste.csv',dtype={'CD_SETOR':str})
assert len(f)==26416 and f.CD_SETOR.is_unique and len(p)==5766
assert np.allclose(f.pct_faces_sem_arvores, 100*f.V05430/f.V05400,atol=1e-6)
assert p.CD_SETOR.is_unique and set(p.CD_SETOR).issubset(set(f.CD_SETOR))
base=pd.read_csv(ROOT/'data/alocacao_otimizada_hipotetica.csv')
c=base.pontos_hipoteticos_por_unidade.to_numpy(float)
cost=base.custo_unitario_hipotetico_RL.to_numpy(float)
cap=base.capacidade_hipotetica.to_numpy(int)
result=milp(c=-c,integrality=np.ones(len(c)),bounds=Bounds(np.zeros(len(c)),cap),constraints=LinearConstraint(cost.reshape(1,-1),-np.inf,[1_000_000]))
assert result.success
assert (np.rint(result.x).astype(int)==base.unidades_selecionadas.to_numpy()).all()
print('OK: 26.416 setores, 5.766 predições e carteira-padrão de R$ 1 milhão reproduzida.')
