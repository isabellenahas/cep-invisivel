"""CEP Invisível: MVP CitizenTech, dados observados e hipóteses rigorosamente separados."""
from __future__ import annotations
import io
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import milp, Bounds, LinearConstraint
import streamlit as st
import folium
import branca.colormap as bcm
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
DATA = ROOT
FOCUS = ('Brasilândia', 'Consolação')
COSTS = {'Arborização': 5000, 'Rampa acessível':25000, 'Abrigo de ônibus':100000}
CAPS = {('Brasilândia','Arborização'):80, ('Consolação','Arborização'):25,
        ('Brasilândia','Rampa acessível'):12, ('Consolação','Rampa acessível'):8,
        ('Brasilândia','Abrigo de ônibus'):4, ('Consolação','Abrigo de ônibus'):3}

st.set_page_config(page_title='CEP Invisível | Inteligência Territorial', page_icon='🌳', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.8rem;max-width:1320px}
.hero{background:linear-gradient(120deg,#172B38,#284955);color:white;padding:1.65rem 1.9rem;border-radius:16px;margin-bottom:1rem}
.hero h1{font-size:2.2rem;margin:0 0 .4rem 0;color:white}.hero p{color:#e1e8e9;margin:0}
.stMetric{border:1px solid #e8edf1;border-radius:12px;padding:10px 14px}
.caption-note{font-size:.85rem;color:#64748b}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA/'dataset_ml.csv', dtype={'CD_SETOR':'string'})
    pred = pd.read_csv(DATA/'predicoes_teste.csv',dtype={'CD_SETOR':'string'})
    metrics = pd.read_csv(DATA/'metricas_teste.csv')
    cv = pd.read_csv(DATA/'comparacao_validacao_cruzada.csv')
    district = pd.read_csv(DATA/'indicadores_96_distritos.csv')
    base = pd.read_csv(DATA/'alocacao_otimizada_hipotetica.csv')
    required=['CD_SETOR','distrito','V05400','V05430','pct_faces_sem_arvores']
    assert set(required).issubset(df.columns),'Dataset não possui as colunas necessárias'
    assert len(df)==26416 and df.CD_SETOR.is_unique, 'Dataset não corresponde ao homologado'
    assert np.allclose(df.pct_faces_sem_arvores, 100*df.V05430/df.V05400,atol=1e-6)
    assert pred.CD_SETOR.is_unique and pred.CD_SETOR.isin(df.CD_SETOR).all()
    return df,pred,metrics,cv,district,base

def brl(x):return 'R$ '+f'{x:,.0f}'.replace(',','.')
def pct(x):return f'{x:.2f}%'.replace('.',',')
def csv_bytes(data):return data.to_csv(index=False,encoding='utf-8-sig').encode('utf-8-sig')

def chart_bar(data,x,y,title,labels=None):
    fig=px.bar(data,x=x,y=y,color=x,text=y,labels=labels or {},title=title,
               color_discrete_sequence=['#D45C3A','#294D63'])
    fig.update_traces(texttemplate='%{text:.2f}',textposition='outside')
    fig.update_layout(showlegend=False,plot_bgcolor='white',yaxis_range=[0,max(data[y])*1.2 if len(data) else 100],margin=dict(t=60,b=30))
    return fig

@st.cache_data(show_spinner=False)
def parse_geo(archive: bytes):
    """Read only validated output features; never infer or invent geographic polygons."""
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        members = z.namelist()
        target=next((n for n in members if n.endswith('setores_dois_distritos_mapa.geojson')),None)
        context=next((n for n in members if n.endswith('municipio_sp_contexto.geojson')),None)
        if target is None or context is None:
            raise ValueError('ZIP precisa conter setores_dois_distritos_mapa.geojson e municipio_sp_contexto.geojson (Etapa 1.4 v4).')
        if z.getinfo(target).file_size>30_000_000 or z.getinfo(context).file_size>10_000_000:
            raise ValueError('GeoJSON acima do limite de segurança')
        focus=json.loads(z.read(target)); city=json.loads(z.read(context))
    if focus.get('type')!='FeatureCollection' or city.get('type')!='FeatureCollection':
        raise ValueError('GeoJSON inválido')
    observed=focus.get('features',[])
    ids=[str(f.get('properties',{}).get('CD_SETOR','')) for f in observed]
    if len(observed)!=740 or len(ids)!=len(set(ids)):
        raise ValueError('Esperados exatamente 740 setores únicos nos dois distritos')
    if set(f.get('properties',{}).get('distrito') for f in observed)!=set(FOCUS):
        raise ValueError('Distritos não correspondem ao recorte aprovado')
    if not city.get('features'):
        raise ValueError('Contorno do município está vazio')
    return focus,city

def render_map(focus,city,pred,kind='deficit'):
    fmap=folium.Map(location=[-23.57,-46.64],tiles=None,zoom_start=10,control_scale=True)
    fmap.get_root().header.add_child(__import__('branca').element.Element('<style>.leaflet-container{background:#ffffff!important}</style>'))
    folium.GeoJson(city,name='Município de São Paulo',style_function=lambda _: {'fillColor':'#e3e7eb','fillOpacity':.9,'color':'#b4c0c8','weight':1}).add_to(fmap)
    if kind=='deficit':
        scale=bcm.linear.YlOrRd_09.scale(0,100)
        scale.caption='Faces sem árvores (%) — Censo 2022'
    else:
        val=pd.to_numeric(pred.loc[pred.distrito.isin(FOCUS),'residuo_observado_menos_previsto']).abs().quantile(.95)
        val=max(30,float(val))
        scale=bcm.LinearColormap(['#2166ac','#f7f7f7','#b2182b'],vmin=-val,vmax=val)
        scale.caption='Resíduo observado − previsto (p.p.) — teste'
    pred_lookup=pred.set_index('CD_SETOR').to_dict('index')
    chosen=[]
    for feat in focus['features']:
        f=json.loads(json.dumps(feat))
        p=f['properties']; sid=str(p['CD_SETOR'])
        if kind=='residuo':
            if sid not in pred_lookup:continue
            p['residuo']=float(pred_lookup[sid]['residuo_observado_menos_previsto'])
            p['previsto']=float(pred_lookup[sid]['previsao_escolhida_bruta'])
        else:
            p['deficit']=float(p['pct_faces_sem_arvores'])
        chosen.append(f)
    keys=['distrito','CD_SETOR','deficit','V05400'] if kind=='deficit' else ['distrito','CD_SETOR','residuo','previsto']
    labels=['Distrito','Setor','Faces sem árvores (%)','Faces pesquisadas'] if kind=='deficit' else ['Distrito','Setor','Erro (p.p.)','Previsto (%)']
    for district in FOCUS:
        subset={'type':'FeatureCollection','features':[f for f in chosen if f['properties']['distrito']==district]}
        attr='deficit' if kind=='deficit' else 'residuo'
        folium.GeoJson(subset,name=district,
            style_function=lambda f: {'fillColor':scale(float(f['properties'][attr])), 'fillOpacity':.8,'color':'#5a6570','weight':.35},
            highlight_function=lambda _: {'fillOpacity':.98,'color':'#1c2730','weight':2},
            tooltip=folium.GeoJsonTooltip(fields=keys,aliases=labels,localize=True,sticky=False)).add_to(fmap)
    # Geographic extent from the verified municipal polygon, not external tiles
    positions=[]
    def walk(coords):
        if isinstance(coords,list):
            if len(coords)>=2 and isinstance(coords[0],(float,int)) and isinstance(coords[1],(float,int)):
                positions.append((float(coords[1]),float(coords[0])))
            else:
                for part in coords:walk(part)
    for f in city['features']:walk(f['geometry']['coordinates'])
    if positions:
        lat=[p[0] for p in positions];lon=[p[1] for p in positions]
        fmap.fit_bounds([[min(lat),min(lon)],[max(lat),max(lon)]])
    scale.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    components.html(fmap.get_root().render(),height=610,scrolling=False)


def build_candidates(df, weight):
    obs=(df[df.distrito.isin(FOCUS)].groupby('distrito')
        .agg(faces=('V05400','sum'),sem=('V05430','sum')))
    obs['deficit']=100*obs['sem']/obs['faces']
    rows=[]
    for (d,t),cap in CAPS.items():
        deficit=float(obs.loc[d,'deficit'])
        pts=(.30+2*(deficit/100)*weight) if t=='Arborização' else (4.0 if t=='Rampa acessível' else 12.0)
        rows.append({'distrito':d,'intervencao':t,'custo_unitario_RL':COSTS[t],
                     'capacidade':cap,'pontos_por_unidade':pts,'deficit_observado_pct':deficit,
                     'base_da_demanda':'IBGE — arborização' if t=='Arborização' else 'Não medida — premissa hipotética'})
    return pd.DataFrame(rows)

def solve(candidates,budget):
    costs=candidates.custo_unitario_RL.to_numpy(float)
    gains=candidates.pontos_por_unidade.to_numpy(float)
    caps=candidates.capacidade.to_numpy(int)
    opt=milp(c=-gains,integrality=np.ones(len(gains)),bounds=Bounds(np.zeros(len(gains)),caps),
             constraints=LinearConstraint(costs.reshape(1,-1),-np.inf,np.array([budget])),options={'time_limit':30})
    if not opt.success or opt.x is None:raise RuntimeError(f'Otimização não concluída: {opt.message}')
    q=np.rint(opt.x).astype(int)
    assert np.all(q>=0) and np.all(q<=caps)
    out=candidates.copy();out['unidades']=q
    out['gasto_RL']=out.unidades*out.custo_unitario_RL
    out['pontos_hipoteticos']=out.unidades*out.pontos_por_unidade
    assert out.gasto_RL.sum()<=budget
    return out


df,pred,metrics,cv,district,base=load_data()
st.markdown('''<div class="hero"><h1>CEP Invisível</h1><p>O endereço não deveria definir seu futuro · Inteligência territorial e decisão pública baseada em evidências</p></div>''',unsafe_allow_html=True)
st.sidebar.markdown('### Explorar o projeto')
page=st.sidebar.radio('Seção', ['Visão geral','Mapa territorial','Machine Learning','Simulador','Metodologia e fontes'],label_visibility='collapsed')
st.sidebar.caption('São Paulo · Censo 2022 · Etapas 1.2–1.5')
st.sidebar.info('Dados observados ≠ previsões ≠ pontuações hipotéticas. O protótipo não estima efeitos causais.')

obs=(df[df.distrito.isin(FOCUS)].groupby('distrito',as_index=False)
 .agg(setores=('CD_SETOR','size'),faces=('V05400','sum'),sem=('V05430','sum'),populacao=('populacao_total','sum')))
obs['deficit_pct']=obs['sem']/obs['faces']*100
obs=obs.sort_values('distrito')

if page=='Visão geral':
    st.subheader('Diagnóstico territorial observado')
    a,b,c,d=st.columns(4)
    a.metric('Setores na base',f'{len(df):,}'.replace(',','.'))
    b.metric('Distritos em São Paulo',df.distrito.nunique())
    c.metric('Brasilândia — déficit',pct(obs.set_index('distrito').loc['Brasilândia','deficit_pct']))
    d.metric('Consolação — déficit',pct(obs.set_index('distrito').loc['Consolação','deficit_pct']))
    left,right=st.columns([1.35,1])
    with left:
        st.plotly_chart(chart_bar(obs,'distrito','deficit_pct','Faces sem árvores por distrito', {'deficit_pct':'Faces sem árvores (%)'}),use_container_width=True)
    with right:
        st.markdown('#### Leitura dos resultados')
        st.write('A Brasilândia apresenta aproximadamente 70,10% de faces pesquisadas sem árvores; na Consolação, 9,37%. Os resultados são agregados ponderados pelo número de faces, não pela população.')
        st.dataframe(obs.rename(columns={'sem':'faces_sem_arvores','faces':'faces_pesquisadas'}),hide_index=True,use_container_width=True)
    st.markdown('#### Diferenças de interpretação')
    st.write('**Diagnóstico:** valores observados no Censo 2022. **ML:** estimativas estatísticas de arborização em setores mantidos fora do treino. **Simulador:** alocação ilustrativa baseada em custos, capacidades e pontuações hipotéticos.')
    st.download_button('Exportar comparação observada',csv_bytes(obs),'comparacao_observada.csv','text/csv')

elif page=='Mapa territorial':
    st.subheader('Mapa territorial — São Paulo')
    st.caption('Município inteiro em cinza claro; somente Brasilândia e Consolação em destaque. Mapas sem servidor externo de tiles.')
    st.markdown('**Carregue o ZIP da Etapa 1.4 v4** (arquivo `CEP_Invisivel_1_4_resultados.zip` gerado no seu Colab).')
    geo_path=DATA/'setores_dois_distritos_mapa.geojson';city_path=DATA/'municipio_sp_contexto.geojson'
    if geo_path.exists() and city_path.exists():
        focus=json.loads(geo_path.read_text(encoding='utf-8'));city=json.loads(city_path.read_text(encoding='utf-8'))
    else:
        uploaded=st.file_uploader('ZIP geográfico homologado',type=['zip'],key='geo')
        if uploaded is None:
            st.warning('Os dados tabulares estão incluídos no pacote. Os mapas precisam do ZIP geográfico da Etapa 1.4, que não foi disponibilizado nesta sessão. Sem esse arquivo não serão exibidas geometrias inventadas.')
            st.stop()
        try:focus,city=parse_geo(uploaded.getvalue())
        except (ValueError,zipfile.BadZipFile,KeyError,TypeError) as e:
            st.error(f'Não foi possível validar o ZIP: {e}');st.stop()
    choice=st.radio('Camada temática',['Déficit observado','Resíduos do ML'],horizontal=True)
    render_map(focus,city,pred,'deficit' if choice=='Déficit observado' else 'residuo')
    st.caption('Resíduos = observado menos previsto. Valores positivos indicam subestimação. As previsões são do conjunto de teste.')

elif page=='Machine Learning':
    st.subheader('Avaliação supervisionada — conjunto de teste')
    st.caption('Modelos já treinados na Etapa 1.3; o aplicativo apresenta resultados e não refaz o treinamento.')
    st.dataframe(metrics,hide_index=True,use_container_width=True)
    st.markdown('**Validação cruzada no treino**')
    st.dataframe(cv,hide_index=True,use_container_width=True)
    p=pred[pred.distrito.isin(FOCUS)].copy()
    fig=px.scatter(p,x='pct_faces_sem_arvores',y='previsao_escolhida_bruta',color='distrito',opacity=.55,
        labels={'pct_faces_sem_arvores':'Observado (%)','previsao_escolhida_bruta':'Previsto (%)'},
        title='Observado × previsto — setores de teste')
    fig.add_shape(type='line',x0=0,y0=0,x1=100,y1=100,line={'dash':'dash','color':'#64748b'})
    fig.update_layout(xaxis_range=[0,100],yaxis_range=[0,100])
    st.plotly_chart(fig,use_container_width=True)
    per=(p.groupby('distrito',as_index=False).agg(MAE_pp=('erro_absoluto','mean'),
        vies_medio_pp=('residuo_observado_menos_previsto','mean'),setores=('CD_SETOR','size')))
    st.dataframe(per,hide_index=True,use_container_width=True)
    st.warning('A Random Forest superestima parte dos setores com déficit baixo e subestima déficits extremos. O erro é maior na Brasilândia; não utilizar predições isoladas como critério de investimento.')

elif page=='Simulador':
    st.subheader('Simulador orçamentário — hipóteses, não impacto comprovado')
    st.caption('Custos e capacidades foram definidos para a demonstração, sem inventário validado de rampas ou abrigos. Somente arborização utiliza déficit local observado.')
    col1,col2=st.columns(2)
    budget=col1.slider('Orçamento hipotético (R$)',100_000,2_000_000,1_000_000,step=25_000,format='%d')
    weight=col2.slider('Peso hipotético da arborização',0.0,3.0,1.0,step=.25)
    candidates=build_candidates(df,weight)
    allocation=solve(candidates,budget)
    spent=int(allocation.gasto_RL.sum())
    m1,m2,m3=st.columns(3)
    m1.metric('Orçamento',brl(budget));m2.metric('Alocado',brl(spent));m3.metric('Saldo',brl(budget-spent))
    view=allocation.pivot(index='distrito',columns='intervencao',values='gasto_RL').fillna(0).reset_index()
    melted=view.melt(id_vars='distrito',var_name='Intervenção',value_name='Gasto (R$)')
    fig=px.bar(melted,x='distrito',y='Gasto (R$)',color='Intervenção',barmode='stack',title='Carteira calculada — valores hipotéticos')
    st.plotly_chart(fig,use_container_width=True)
    cols=['distrito','intervencao','unidades','gasto_RL','pontos_hipoteticos','base_da_demanda']
    st.dataframe(allocation[cols],hide_index=True,use_container_width=True)
    st.metric('Pontuação normativa ilustrativa',f'{allocation.pontos_hipoteticos.sum():.2f}')
    if budget==1_000_000 and weight==1.0:
        check=allocation.sort_values(['distrito','intervencao']).reset_index(drop=True)
        target=base[['distrito','intervencao','unidades_selecionadas']].sort_values(['distrito','intervencao']).reset_index(drop=True)
        assert check.unidades.tolist()==target.unidades_selecionadas.tolist(), 'A solução padrão divergiu dos resultados homologados'
        st.success('Configuração padrão reproduz a alocação da Etapa 1.5.')
    st.download_button('Exportar carteira do cenário',csv_bytes(allocation),'carteira_hipotetica.csv','text/csv')
    st.info('A maximização usa pontos hipotéticos por unidade. Alterar o peso não comprova maior benefício social e não constitui estimativa causal ou recomendação de política pública.')

else:
    st.subheader('Metodologia e transparência')
    st.markdown('''**Fonte:** IBGE Censo Demográfico 2022 — agregados setoriais de entorno, população, renda e malha geográfica.\n
**Unidade:** setor censitário, com comparação agregada por faces pesquisadas.\n
**Alvo supervisionado:** `100 × V05430 / V05400` (faces sem árvores / faces pesquisadas).\n
**Experimento:** Ridge e Random Forest, separação por distritos, validação cruzada por grupos exclusivamente no treino.\n
**Conjunto de teste:** inclui integralmente Brasilândia e Consolação, além de outros 19 distritos.\n
**Limitações:** corte transversal de 2022; ausência de inventários reais de rampas/abrigos; desempenho preditivo moderado; modelo não identifica efeito causal.\n
**Escopo realmente implementado:** arborização, diagnóstico, avaliação de modelos e alocação hipotética. Saúde, educação, transporte efetivo e calor urbano não possuem indicadores integrados nesta versão.\n
**Arquivos e integridade:** os CSVs homologados estão incluídos no pacote para permitir auditoria e reprodução. O pacote geográfico de 1.4 deve ser fornecido separadamente.''')
    st.download_button('Baixar dicionário básico de dados', (ROOT/'DICIONARIO.md').read_bytes(), 'DICIONARIO.md','text/markdown')
    st.caption('Projeto acadêmico CitizenTech. Sem inferência causal e sem autorização para uso operacional em decisões públicas.')
