# CEP Invisível — Etapa 1.6 (Streamlit)

## Execução local (Windows, Mac ou Linux)

Requer Python 3.10–3.12.

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

O aplicativo abrirá no navegador (normalmente http://localhost:8501).

## Publicar

Se desejar publicar no Streamlit Community Cloud, envie o conteúdo desta pasta para um repositório GitHub e indique `app.py` como arquivo principal. **Revise antes de tornar o repositório público.**

## Mapas

Os dados tabulares homologados já estão incluídos. Por segurança metodológica, o ZIP com os GeoJSONs finais da Etapa 1.4 **não está incluído**, pois não foi recebido nesta sessão. Na seção **Mapa territorial**, envie `CEP_Invisivel_1_4_resultados.zip` produzido pelo seu Colab v4, contendo:

- `setores_dois_distritos_mapa.geojson`
- `municipio_sp_contexto.geojson`

Alternativamente, coloque esses dois arquivos em `data/` para ativar os mapas automaticamente. Nenhum serviço de tiles é utilizado; o JavaScript/CSS padrão do Folium poderá precisar de internet no navegador.

## Fontes, limites e estado

- Dados de setores do IBGE Censo 2022, 26.416 registros homologados.
- Previsões fora do treino: 5.766 setores; comparativos históricos da Etapa 1.3.
- A simulação usa `scipy.optimize.milp` com quantidades inteiras e orçamento limitado.
- Custos unitários: árvores R$ 5 mil; rampas R$ 25 mil; abrigos R$ 100 mil, todos hipotéticos.
- Pontuação: árvores = 0.3 + 2*(déficit distrital/100)*peso; rampas = 4; abrigos = 12. **Hipóteses normativas, não impactos reais.**
- Aplicação não treina modelos, não utiliza serviços pagos e não depende de APIs externas de mapas.

## Validação

```bash
python smoke_test.py
```

O teste verifica unicidade do dataset, integridade do alvo, compatibilidade das previsões e solução padrão do simulador.
