# Dicionário resumido — CEP Invisível

- `CD_SETOR`: identificador de 15 dígitos do setor censitário.
- `distrito`: distrito administrativo de São Paulo.
- `V05400`: total de faces de quadra pesquisadas no entorno (denominador).
- `V05430`: faces de quadra sem árvores (numerador).
- `pct_faces_sem_arvores`: 100 x V05430 / V05400.
- `populacao_total`: número de moradores.
- `domicilios_ocupados`: total de domicílios particulares ocupados.
- `densidade_hab_km2`: moradores por quilômetro quadrado.
- `renda_responsavel_media`: rendimento médio nominal de responsáveis com rendimento; não renda per capita.
- `previsao_escolhida_bruta`: predição de Random Forest do conjunto de teste.
- `residuo_observado_menos_previsto`: percentual observado menos previsto, em pontos percentuais.
- `pontos_hipoteticos`: pontuação normativa *ilustrativa*, sem equivalência a impactos reais.
