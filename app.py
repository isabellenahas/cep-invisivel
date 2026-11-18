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
DATA = ROOT / 'data' if (ROOT / 'data' / 'dataset_ml.csv').exists() else ROOT
FOCUS = ('Brasilândia', 'Consolação')
COSTS = {'Arborização': 5000, 'Rampa acessível':25000, 'Abrigo de ônibus':100000}
CAPS = {('Brasilândia','Arborização'):80, ('Consolação','Arborização'):25,
        ('Brasilândia','Rampa acessível'):12, ('Consolação','Rampa acessível'):8,
        ('Brasilândia','Abrigo de ônibus'):4, ('Consolação','Abrigo de ônibus'):3}

st.set_page_config(page_title='CEP Invisível | Inteligência Territorial', page_icon='🌳', layout='wide', initial_sidebar_state='collapsed')
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

def validate_geo(focus,city):
    """Usa apenas a malha homologada de 740 setores e contexto municipal real."""
    if focus.get('type')!='FeatureCollection' or city.get('type')!='FeatureCollection':
        raise ValueError('GeoJSON não é uma FeatureCollection')
    observed=focus.get('features',[])
    ids=[str(f.get('properties',{}).get('CD_SETOR','')) for f in observed]
    if len(observed)!=740 or len(ids)!=len(set(ids)):
        raise ValueError('Esperados 740 setores únicos nos dois distritos')
    if set(f.get('properties',{}).get('distrito') for f in observed)!=set(FOCUS):
        raise ValueError('Distrito fora do recorte aprovado')
    if not city.get('features'):
        raise ValueError('Contorno do município vazio')

@st.cache_data(show_spinner=False)
def parse_geo(archive: bytes):
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        members=z.namelist()
        target=next((n for n in members if n.endswith('setores_dois_distritos_mapa.geojson')),None)
        context=next((n for n in members if n.endswith('municipio_sp_contexto.geojson')),None)
        if not target or not context:
            raise ValueError('ZIP sem os dois GeoJSONs finais da Etapa 1.4 v4')
        if z.getinfo(target).file_size>30_000_000 or z.getinfo(context).file_size>10_000_000:
            raise ValueError('GeoJSON acima do limite de segurança')
        focus=json.loads(z.read(target));city=json.loads(z.read(context))
    validate_geo(focus,city)
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
    components.html(fmap.get_root().render(),height=620,scrolling=False)


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



# ---------------------------------------------------------------------------------
# CEP INVISÍVEL | Interface 2.0 — orientação a produto, sem alterar método científico
# ---------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {--ink:#112a31;--sub:#53646b;--line:#e5e9e7;--paper:#f7f9f8;--teal:#087b72;--mint:#d4efe7;--coral:#f27050}
html, body, [class*="css"], [data-testid="stApp"]{font-family:'DM Sans',sans-serif}
.stApp {background:var(--paper)}
.block-container{max-width:1280px;padding-top:1.2rem;padding-bottom:4rem}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;color:var(--ink)}
h1 {font-weight:800!important}
div[data-testid="stMetric"] {background:white;border:1px solid var(--line);border-radius:17px;padding:15px 18px;box-shadow:0 7px 24px rgba(14,44,37,.035)}
[data-testid="stMetricLabel"] {color:#61736f;font-weight:600}
[data-testid="stMetricValue"] {color:#16382f;font-family:'Manrope',sans-serif;letter-spacing:-.04em}
[data-testid="stFileUploader"],div[data-testid="stDataFrame"] {border-radius:12px}
.stButton button[kind="primary"],.stDownloadButton button[kind="primary"]{background:#087b72;color:white;border-radius:12px;border:0;font-weight:750}
.stButton button,.stDownloadButton button {border-radius:12px;font-weight:600}
[data-testid="stHorizontalBlock"] {gap:1rem}
.hero {position:relative;background:linear-gradient(110deg,#11343a 0%,#0e665e 75%,#138a77 100%);padding:20px 28px;border-radius:18px;color:white;overflow:hidden;margin:12px 0 22px}
.hero::after{content:'';position:absolute;width:165px;height:165px;border:1px solid rgba(255,255,255,.13);border-radius:50%;right:-40px;top:-65px;box-shadow:0 0 0 35px rgba(255,255,255,.035)}
.hero .eyebrow{font-size:10px;font-weight:800;letter-spacing:.13em;color:#99e8d0;text-transform:uppercase;margin-bottom:7px}
.hero h1{color:white!important;font-size:clamp(23px,2.7vw,34px)!important;line-height:1.16;margin:0 0 7px;font-weight:800}
.hero p{font-size:13px;line-height:1.5;max-width:820px;color:#d6ebe6;margin:0}
.hero .tag{display:inline-block;margin-top:10px;border:1px solid #82c6b5;color:#eafff7;border-radius:99px;padding:4px 10px;font-size:10px;font-weight:800}
.section-kicker {font-size:11px;font-weight:800;letter-spacing:.17em;text-transform:uppercase;color:var(--teal);margin:12px 0 4px}
.section-title {font-family:'Manrope';font-size:27px;font-weight:800;letter-spacing:-.04em;color:var(--ink);margin:0 0 8px}
.section-subtitle{font-size:14px;color:var(--sub);line-height:1.75;margin:0 0 22px}
.story-card{background:white;border:1px solid var(--line);border-radius:18px;padding:24px 27px;margin:10px 0 24px}
.story-card h3 {margin:0 0 8px;font-size:20px}
.story-card p{color:var(--sub);line-height:1.75;margin-bottom:0}
.stat-strip{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:12px 0 20px}
.stat-tile {border:1px solid #d8e5df;border-radius:16px;padding:20px 22px;background:#eaf4ee}
.stat-tile.dark{background:#173c3f;color:white;border-color:#173c3f}
.stat-tile .sm{font-size:12px;font-weight:700;opacity:.75}
.stat-tile .num{font-size:32px;letter-spacing:-.045em;font-weight:800;font-family:'Manrope'}
.stat-tile .foot{font-size:12px;opacity:.75;margin-top:5px}
.note {border-left:3px solid #0a8d79;border-radius:0 10px 10px 0;padding:11px 17px;background:#e8f5f0;color:#365d55;font-size:13px;line-height:1.7;margin:14px 0}
.pill{display:inline-block;padding:6px 11px;border:1px solid #a1cfc1;border-radius:100px;color:#0d6255;background:#edf8f2;font-size:11px;font-weight:800;margin-right:6px}
.footer {padding-top:30px;margin-top:30px;border-top:1px solid var(--line);color:#667873;font-size:12px;line-height:1.7}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;background:white;border:1px solid var(--line);border-radius:16px;padding:12px 20px;margin-bottom:10px;box-shadow:0 5px 20px rgba(10,40,35,.035)}
.topbar .brand{display:flex;align-items:center;gap:11px;font-family:'Manrope',sans-serif;font-size:18px;font-weight:800;letter-spacing:-.04em;color:var(--ink)}
.topbar .mark{background:#087b72;color:white;display:grid;place-items:center;width:32px;height:32px;border-radius:11px;font-size:18px}
.topbar .meta{color:#647872;font-size:11px;font-weight:700;letter-spacing:.05em}
.nav-caption{font-size:11px;color:#71837d;font-weight:750;margin:0 0 5px 5px;text-transform:uppercase;letter-spacing:.12em}
/* Streamlit radio styled as a horizontal application-navigation bar */
.st-key-pagina_principal {border:1px solid #e4eae6;background:#fff;padding:7px 9px;border-radius:15px;margin-bottom:8px}
.st-key-pagina_principal [role="radiogroup"] {display:flex;gap:5px;flex-wrap:wrap}
.st-key-pagina_principal label {padding:8px 12px;border-radius:10px;margin:0;font-size:13px;font-weight:650;transition:background .15s}
.st-key-pagina_principal label:has(input:checked) {background:#e3f3ed!important;color:#065f53!important}
.st-key-pagina_principal label:hover {background:#edf4f0}
@media(max-width:750px){.hero{padding:16px 18px}.hero h1{font-size:24px!important}.topbar{padding:10px 12px}.topbar .meta{display:none}.stat-strip{grid-template-columns:1fr}.block-container{padding-top:1rem}}
</style>
"""
st.markdown(CSS,unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def observed_by_district(frame):
    out=(frame[frame.distrito.isin(FOCUS)].groupby('distrito',as_index=False)
       .agg(setores=('CD_SETOR','size'),faces=('V05400','sum'),sem=('V05430','sum'),populacao=('populacao_total','sum')))
    out['deficit_pct'] = 100 * out['sem'] / out['faces']
    return out.sort_values('distrito').reset_index(drop=True)

def heading(kicker,title,subtitle=''):
    st.markdown(f'<div class="section-kicker">{kicker}</div><div class="section-title">{title}</div><p class="section-subtitle">{subtitle}</p>',unsafe_allow_html=True)

def plot_theme(fig,height=390):
    fig.update_layout(height=height,font_family='DM Sans',font_color='#26413c',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=15,r=15,t=35,b=25),legend_title_text='',
                      xaxis=dict(showgrid=False,zeroline=False),yaxis=dict(gridcolor='#e2eae7',zeroline=False))
    return fig

def section_footer():
    st.markdown('<div class="footer"><b>CEP Invisível · FIAP Global Solution 2026</b> · Dados oficiais IBGE Censo 2022 · Protótipo CitizenTech. Resultados observados, previsões e simulações representam conceitos diferentes. Nenhuma estimativa causal de impacto é apresentada.</div>',unsafe_allow_html=True)

# Navegação principal horizontal: sem menu lateral, com aparência de produto digital.
pages=['Panorama','Explorar territórios','Inteligência preditiva','Laboratório de cenários','Como funciona']
st.markdown("""<div class="topbar"><div class="brand"><span class="mark">◈</span>CEP INVISÍVEL</div>
<div class="meta">INTELIGÊNCIA TERRITORIAL &nbsp;·&nbsp; SÃO PAULO &nbsp;·&nbsp; CENSO 2022</div></div>""",unsafe_allow_html=True)
page=st.radio('Navegação',pages,horizontal=True,label_visibility='collapsed',key='pagina_principal')

df,pred,metrics,cv,district,base=load_data()
obs=observed_by_district(df)
ob=obs.set_index('distrito')
bra=float(ob.loc['Brasilândia','deficit_pct']); con=float(ob.loc['Consolação','deficit_pct'])

if page=='Panorama':
    st.markdown("""<div class="hero"><div class="eyebrow">CitizenTech · Protótipo acadêmico</div>
    <h1>O endereço não deveria definir seu futuro.</h1>
    <p>Inteligência territorial para enxergar desigualdades, avaliar previsões e explorar prioridades com transparência.</p>
    <span class="tag">DADOS REAIS DO IBGE · 2022</span></div>""",unsafe_allow_html=True)

if page=='Panorama':
    heading('01 / O DIAGNÓSTICO','Duas regiões. Realidades diferentes.',
            'Um olhar comparativo sobre o déficit de arborização urbana, calculado a partir das faces de quadra pesquisadas pelo IBGE.')
    a,b,c,d=st.columns(4)
    a.metric('Setores analisados',f'{len(df):,}'.replace(',','.'))
    b.metric('Distritos em São Paulo',df.distrito.nunique())
    c.metric('Brasilândia · sem árvores',pct(bra))
    d.metric('Consolação · sem árvores',pct(con))
    left,right=st.columns([1.25,.85],gap='large')
    with left:
        fig=px.bar(obs,x='distrito',y='deficit_pct',color='distrito',text='deficit_pct',
            color_discrete_map={'Brasilândia':'#df7752','Consolação':'#167d75'},
            labels={'distrito':'','deficit_pct':'Faces sem árvores (%)'})
        fig.update_traces(texttemplate='%{text:.2f}%',textposition='outside',marker_line_width=0)
        fig.update_layout(showlegend=False,yaxis_range=[0,90])
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    with right:
        st.markdown(f"""<div class="story-card"><div class="section-kicker">O que os dados mostram</div><h3>{bra-con:.2f} pontos percentuais</h3>
        <p>É a diferença entre os déficits agregados observados de arborização na Brasilândia e na Consolação.
        O indicador é ponderado pelo número de faces pesquisadas, não pelo total de moradores.</p></div>""",unsafe_allow_html=True)
        st.markdown('<div class="note">A arborização é o indicador territorial implementado com dados reais. O MVP não mede diretamente calor urbano ou acesso a equipamentos públicos.</div>',unsafe_allow_html=True)
    heading('Explorar os dados','O que entra na comparação','Total de faces pesquisadas, faces sem árvores e população residente nos setores elegíveis.')
    st.dataframe(obs.rename(columns={'distrito':'Distrito','setores':'Setores','faces':'Faces pesquisadas','sem':'Faces sem árvores','populacao':'População nos setores','deficit_pct':'Déficit (%)'}),hide_index=True,use_container_width=True)
    st.download_button('Baixar comparação em CSV',csv_bytes(obs),'comparacao_territorial.csv','text/csv')

elif page=='Explorar territórios':
    heading('02 / GEOGRAFIA','Onde a desigualdade aparece','O município é exibido como contexto; apenas Brasilândia e Consolação possuem setores destacados e dados observados nesta visualização.')
    # GitHub: colocar os dois GeoJSONs na raiz, ao lado de app.py, ou na pasta data/.
    local_dirs=[DATA, ROOT, ROOT/'data']
    local_dir=next((p for p in local_dirs if (p/'setores_dois_distritos_mapa.geojson').is_file()
                    and (p/'municipio_sp_contexto.geojson').is_file()),None)
    if local_dir is not None:
        try:
            focus=json.loads((local_dir/'setores_dois_distritos_mapa.geojson').read_text(encoding='utf-8'))
            city=json.loads((local_dir/'municipio_sp_contexto.geojson').read_text(encoding='utf-8'))
            validate_geo(focus,city)
        except (ValueError,json.JSONDecodeError) as e:
            st.error(f'Arquivos geográficos locais inválidos: {e}');st.stop()
        st.caption('Fonte geográfica oficial carregada automaticamente · IBGE Censo 2022')
    else:
        # Caminho ainda mais simples no GitHub: enviar apenas o ZIP final para a raiz.
        archive_candidates=[folder/name for folder in [DATA,ROOT,ROOT/'data']
            for name in ('CEP_Invisivel_1_4_resultados.zip','CEP_Invisivel_1_4_resultados (1).zip')]
        geo_archive=next((p for p in archive_candidates if p.is_file()),None)
        if geo_archive is not None:
            try:focus,city=parse_geo(geo_archive.read_bytes())
            except (ValueError,zipfile.BadZipFile,KeyError,TypeError) as e:
                st.error(f'ZIP geográfico publicado no GitHub é inválido: {e}');st.stop()
            st.caption('Mapa carregado automaticamente do pacote geográfico homologado · IBGE Censo 2022')
        else:
            st.info('Para disponibilizar o mapa automaticamente a todos os visitantes, envie '
                    '`CEP_Invisivel_1_4_resultados.zip` ao GitHub na mesma pasta de `app.py`. '
                    'Alternativamente, publique os dois GeoJSONs extraídos do ZIP. '
                    'Não é necessário modificar o código novamente.')
            uploaded=st.file_uploader('Teste temporário: ZIP geográfico da Etapa 1.4',type=['zip'],key='geo')
            if uploaded is None:st.stop()
            try:focus,city=parse_geo(uploaded.getvalue())
            except (ValueError,zipfile.BadZipFile,KeyError,TypeError) as e:
                st.error(f'O ZIP não passou na validação: {e}');st.stop()
    choice=st.radio('Indicador exibido',['Déficit observado','Erro do modelo'],horizontal=True)
    render_map(focus,city,pred,'deficit' if choice=='Déficit observado' else 'residuo')
    st.markdown('<div class="note">Mapa interativo sem mosaicos externos. Resíduo positivo significa que o modelo subestimou o déficit observado. Os dois distritos estão no conjunto de teste.</div>',unsafe_allow_html=True)

elif page=='Inteligência preditiva':
    heading('03 / MACHINE LEARNING','Onde a previsão ajuda — e onde falha','Comparação supervisionada com validação cruzada por distrito, sem reutilizar o conjunto de teste para escolher o melhor modelo.')
    m=metrics.set_index('modelo')
    rf=m.loc['Random Forest']
    a,b,c=st.columns(3)
    a.metric('MAE · Random Forest',f'{rf.MAE_pp:.2f} p.p.'.replace('.',','))
    b.metric('RMSE · Random Forest',f'{rf.RMSE_pp:.2f} p.p.'.replace('.',','))
    c.metric('R² no conjunto de teste',f'{rf.R2:.3f}'.replace('.',','))
    st.markdown('#### Comparação com os modelos de referência')
    fig=px.bar(metrics,x='modelo',y='MAE_pp',color='modelo',text='MAE_pp',
               color_discrete_map={'Random Forest':'#087b72','Ridge':'#6b9f97','Baseline média':'#e7a08e'},
               labels={'modelo':'Modelo','MAE_pp':'Erro absoluto médio (p.p.)'})
    fig.update_traces(texttemplate='%{text:.2f}',textposition='outside')
    fig.update_layout(showlegend=False,yaxis_range=[0,33])
    st.plotly_chart(plot_theme(fig,320),use_container_width=True)
    st.markdown('#### Observado × previsto: os dois territórios demonstrativos')
    p=pred[pred.distrito.isin(FOCUS)].copy()
    fig=px.scatter(p,x='pct_faces_sem_arvores',y='previsao_escolhida_bruta',color='distrito',opacity=.46,
         color_discrete_map={'Brasilândia':'#df7752','Consolação':'#087b72'},
         hover_data=['CD_SETOR'],labels={'pct_faces_sem_arvores':'Déficit observado (%)','previsao_escolhida_bruta':'Déficit previsto (%)','distrito':'Distrito'})
    fig.add_shape(type='line',x0=0,y0=0,x1=100,y1=100,line=dict(dash='dash',color='#849592'))
    fig.update_layout(xaxis_range=[0,100],yaxis_range=[0,100])
    st.plotly_chart(plot_theme(fig,490),use_container_width=True)
    err=(p.groupby('distrito',as_index=False).agg(MAE_pp=('erro_absoluto','mean'),vies_pp=('residuo_observado_menos_previsto','mean'),setores=('CD_SETOR','size')))
    st.dataframe(err,hide_index=True,use_container_width=True)
    with st.expander('Ver validação cruzada e hiperparâmetros'):
        st.dataframe(cv,hide_index=True,use_container_width=True)
    st.markdown('<div class="note"><b>Limitação importante:</b> o modelo tem desempenho moderado (R² ≈ 0,345) e tende a subestimar déficits altos, inclusive na Brasilândia. Não é uma estimativa causal nem substitui o indicador observado.</div>',unsafe_allow_html=True)

elif page=='Laboratório de cenários':
    heading('04 / SIMULAÇÃO','E se tivéssemos R$ 1 milhão?','Um laboratório transparente de escolhas com orçamento limitado, quantidades inteiras e premissas hipotéticas — não uma recomendação de investimento público.')
    left,right=st.columns([1,1],gap='large')
    with left:
        budget=st.slider('Orçamento disponível (R$)',100_000,2_000_000,1_000_000,step=25_000,format='%d')
    with right:
        weight=st.slider('Importância relativa da arborização',0.0,3.0,1.0,step=.25,help='Parâmetro normativo do cenário; não mede causalidade.')
    allocation=solve(build_candidates(df,weight),budget)
    spent=int(allocation.gasto_RL.sum())
    a,b,c=st.columns(3)
    a.metric('Orçamento',brl(budget));b.metric('Alocado',brl(spent));c.metric('Saldo',brl(budget-spent))
    view=allocation.pivot(index='distrito',columns='intervencao',values='gasto_RL').fillna(0).reset_index()
    melted=view.melt(id_vars='distrito',var_name='Intervenção',value_name='Gasto (R$)')
    fig=px.bar(melted,x='distrito',y='Gasto (R$)',color='Intervenção',barmode='stack',
      color_discrete_map={'Arborização':'#087b72','Rampa acessível':'#df7752','Abrigo de ônibus':'#6d8bc6'},
      labels={'distrito':'Distrito'})
    st.plotly_chart(plot_theme(fig,410),use_container_width=True)
    st.markdown('#### A carteira calculada')
    st.dataframe(allocation[['distrito','intervencao','unidades','gasto_RL','pontos_hipoteticos','base_da_demanda']],hide_index=True,use_container_width=True)
    st.metric('Pontuação normativa ilustrativa',f'{allocation.pontos_hipoteticos.sum():.2f}'.replace('.',','))
    if budget==1_000_000 and weight==1.0:
        check=allocation.sort_values(['distrito','intervencao']).reset_index(drop=True)
        target=base[['distrito','intervencao','unidades_selecionadas']].sort_values(['distrito','intervencao']).reset_index(drop=True)
        if check.unidades.tolist()==target.unidades_selecionadas.tolist():
            st.success('Carteira de referência da Etapa 1.5 reproduzida.')
        else:st.error('A carteira divergiu dos resultados homologados. Verifique os arquivos de entrada.')
    st.download_button('Exportar alocação deste cenário',csv_bytes(allocation),'carteira_hipotetica.csv','text/csv')
    st.markdown('<div class="note"><b>O que é real?</b> O déficit observado de arborização. <b>O que é hipótese?</b> Custos, capacidades, prioridades de rampas/abrigos e pontos por unidade. Alterar os controles não prevê benefícios reais.</div>',unsafe_allow_html=True)

else:
    heading('05 / TRANSPARÊNCIA','Ciência antes da narrativa','A proposta só é útil quando é possível distinguir evidência observada, previsão estatística e hipótese de decisão.')
    with st.container(border=True):
        st.markdown('#### Fontes e rastreabilidade')
        st.markdown('**Fonte:** IBGE Censo Demográfico 2022 — agregados setoriais de entorno, população, renda e malha geográfica. **Unidade:** setor censitário. **Alvo:** `100 × V05430 / V05400`.')
        st.markdown('**Dataset:** 26.416 setores elegíveis de São Paulo. **Recorte demonstrativo:** Brasilândia e Consolação.')
        st.markdown('**Machine Learning:** Ridge e Random Forest, separação por distritos, GroupKFold apenas no treino; MAE, RMSE e R² no conjunto de teste.')
        st.markdown('**Simulador:** otimização inteira com custos, capacidades e pontuações normativas hipotéticas. Não mede efeitos causais.')
    with st.expander('Riscos, vieses e limitações',expanded=True):
        st.markdown("""- A medição é transversal, referente a **2022**, e não representa previsão temporal.
- Setores com poucas faces pesquisadas possuem maior variabilidade da proporção observada.
- Os modelos têm erros expressivos nos extremos e podem subestimar territórios com déficit alto.
- A proximidade espacial e a representatividade do conjunto de teste limitam a generalização.
- O MVP não integra indicadores validados de saúde, educação, transporte efetivo ou calor urbano.
- Custos e pontuações de cenários são hipóteses, sem comprovação de impacto público.""")
    st.markdown('#### Arquivos disponíveis para auditoria')
    st.download_button('Baixar dicionário de dados (CSV)',csv_bytes(df.head(0)),'campos_dataset.csv','text/csv')
    st.markdown('**Repositório e execução:** o projeto técnico original inclui notebooks executados no Google Colab. O Streamlit exibe resultados e não refaz o treinamento.')

section_footer()
