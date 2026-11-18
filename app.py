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
    geo_path=DATA/'setores_dois_distritos_mapa.geojson'; city_path=DATA/'municipio_sp_contexto.geojson'
    if geo_path.is_file() and city_path.is_file():
        try:
            focus=json.loads(geo_path.read_text(encoding='utf-8'))
            city=json.loads(city_path.read_text(encoding='utf-8'))
            if len(focus.get('features',[]))!=740 or not city.get('features'):
                raise ValueError('GeoJSONs locais não correspondem aos dados homologados')
        except (ValueError,json.JSONDecodeError) as e:
            st.error(f'Arquivos geográficos inválidos: {e}');st.stop()
        st.success('Geometrias locais carregadas. Os visitantes não precisam enviar arquivos.')
    else:
        with st.container(border=True):
            st.markdown('**Carregar os mapas homologados**')
            st.caption('Envie `CEP_Invisivel_1_4_resultados.zip`, produzido na Etapa 1.4. Nenhuma geometria será simulada.')
            uploaded=st.file_uploader('Arquivo ZIP geográfico',type=['zip'],key='geo')
        if uploaded is None:
            st.info('Os indicadores estão disponíveis. Para visualizar os polígonos, envie o ZIP da Etapa 1.4; se você publicar os dois GeoJSONs junto aos CSVs no GitHub, eles aparecerão automaticamente para todos.')
            st.stop()
        try: focus,city=parse_geo(uploaded.getvalue())
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
