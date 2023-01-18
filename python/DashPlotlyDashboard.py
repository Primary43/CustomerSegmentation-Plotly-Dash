app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                  meta_tags=[
                      {
                      "name": "viewport",
                      "content": "width=device-width, initial-scale=1, maximum-scale=1",
                      }
                  ])
#color_dict
colors = {
          'main-background':'#eeeeee', #grey
          'background': '#fffff', #white
          'text': '#111111',
          'primary':'#0D0888'
         }
#card-------------------
card_content = [
    dbc.CardHeader("Card header", className="card-title"),
    dbc.CardBody([
        html.P("This is some card content that we'll reuse"
                ,className="card-text",)
    ])
]

#layout------------------------------------------------------------------------------------------------------------
app.layout = dbc.Container([
    #header
    dbc.Row(dbc.Col(html.H1('Customer Segmentation', className='text-center text-primary, mb-3'))),
    html.Hr(), 
    #title box 1
    dbc.Row(dbc.Col(
        html.Div(children=
        '''
        The goal of this customer segmentation is to put customers into groups with similar 
        characteristics, such as gender, age, annual income, and spending score. The segmentation
         includes data exploration, feature engineering, clustering, and analysis.
        ''',
        style={'background-color': 'background',
               'margin': '20px'},
        className='text-center fw-bold'
    ))),
    #graph row1
    dbc.Row([ 
        dbc.Col([  # 1R/1C
            html.H5('Gender Distribution', className='text-center'),
            dcc.Graph(id='graph1',
                      figure=fig2, 
                      style={'height':300})
            ],width={'size': 8, 'offset': 0, 'order': 1}),  
             
         dbc.Col([  # 1R/2C
            html.H5('Gender Ratio', className='text-center'),
            dcc.Graph(id='graph2',
                      figure=fig1,
                      style={'height':300})
            ], width={'size': 4, 'offset': 0, 'order': 2})
    ]),
    #description1
    dbc.Row([ 
        html.Div([
            html.P(
                  """Of the 200 records in the dataset, 56% are female customers and 44% are male. 
                  The mean age of all customers is 38.85 years, with an average annual income of $60.56 and a mean spending score of 50.20.
                   Male customers have a slightly higher mean age (39.80 years) and mean annual income ($59.25) compared to female customers (38.09 years and $51.52). Female customers have a higher mean spending score (51.52) compared to male customers (48.51). 
                  The most common age range among customers is 30-35 years, followed by those under 20 years and those between 45 and 50 years. The majority of customers have an income in the range of $50–62.5, followed by $75–87.5. The most common spending score among customers is 50-60, followed by 40-50.
                  """)
            ],style={
                    "width": "95%",
                    "display": "inline-block",
                    "margin-left": "20px",
                    "margin-right": "20px",
                    "margin-top": "20px",
                    "margin-bottom": "10px",
                    "text-indent": "35px",
                    "text-align": "justify",
                    "font-size":"14px"}
            )
    ]),
    #component1----------------------------------------------------------------
    dbc.Row([
        html.H5('Select Filter'),
        html.Hr(),
        dbc.Col([
            #dropdown gender
            html.H6('Customer Gender:'),
            dcc.Dropdown(
                multi=True,
                id='GenderSelect',
                options=[
                         {'label': x, 'value': x, 'disabled':False}
                         for x in df['Gender'].unique()
                ],
                value=['Female','Male'],
                style={'font-size': 15, 'padding-left': 10}
            ),
            html.Br(),
            #range slider
            html.H6('Age Range:'),
            dcc.RangeSlider(
                df['Age'].min(), df['Age'].max(), 5, 
                value=[18, 70],
                id='AgeSelect'
            ),            
        ],style={'marginLeft': '20px','marginRight': '20px'}),
        dbc.Col([
            html.Br(),
            #income slider
            html.H6('Income Range(K$):'),
            dcc.RangeSlider(
                df['Annual_Income'].min(), df['Annual_Income'].max(), 10, 
                value=[15, 137],
                id='Annual_IncomeSelect'
            ),
            #spending score slider
            html.H6('Spending Score:'),
            dcc.RangeSlider(
                df['Spending_Score'].min(), df['Spending_Score'].max(), 10, 
                value=[1, 99],
                id='Spending_ScoreSelect'
            )
        ],style={'marginLeft': '20px','marginRight': '20px'})
     ],style={'background-color': 'background',
              'paddingBottom': '20px'}),

    #--------------------------------------------------------------------------
    #graph row2
    dbc.Row([ 
         dbc.Col([  # 2R/1C
             html.H5('Scatterplot Matrix of Feature', className='text-center'),
             dcc.Graph(id='graph3',
                       figure=fig3,
                       style={'height':500}),
             ], width={'size': 8, 'offset': 0, 'order': 1}),
         dbc.Col([  # 2R/2C
             html.H5('Feature Correlation', className='text-center'),
             dcc.Graph(id='graph4',
                       figure=fig4,
                       style={'height':500}),
             ], width={'size': 4, 'offset': 0, 'order': 2})
    ]),
    #description2
    dbc.Row([
        html.Div(
             [html.P(
                      """ There is a strong positive correlation between the customer ID
                       and annual income (+0.98) and a negative correlation between age 
                       and spending score (-0.33). The correlation between gender and 
                       spending score is positive (+0.058), but there is no significant 
                       correlation in other features.
                      """
              )],
              style={
                      "width": "95%",
                      "display": "inline-block",
                      "margin-left": "20px",
                      "margin-right": "20px",
                      "margin-top": "20px",
                      "margin-bottom": "10px",
                      "text-indent": "35px",
                      "text-align": "justify",
                      "font-size":"14px"
                      },
        )
    ]),
    html.Hr(),
    #title box2
    dbc.Row([
       html.Div(children=
              '''
              To identify a different customer segment and underlie the similar characteristics 
              of their shopping behavior, an unsupervised learning algorithm will be applied to 
              distinguish clusters using a similarity measurement and the distance of the similarity function.
              ''',
              style={'background-color': 'background',
                    'margin': '20px'},
              className='text-center fw-bold'
              )
    ]),
    #graph3
    dbc.Row([ 
       dbc.Col([  # 3R/1C
              html.H5('DBSCAN Clustering', className='text-center'),
              dcc.Graph(id='graph5',
                          figure=fig5,
                          style={'height':500}),
       ]),
       dbc.Col([  # 3R/1C
              html.H5('K-means Clustering', className='text-center'),
              dcc.Graph(id='graph6',
                          figure=fig6,
                          style={'height':500})
       ])
    ]),
    dbc.Row([ 
        html.Div([
            html.H6('Feature Transformation:'),
            html.P(
                  """
                  As similarity measures can be easily affected by different scales, 
                  we apply Min-Max normalization to handle various magnitudes of numerical values.
                  """),
            html.H6('Dimensional Reduction:'),
            html.P(
                  """
                  To avoid the curse of the high-dimensional data, we use Principal
                  Component Analysis (PCA) with its explained variance ratio to evaluate the usefulness of each
                  number of principal components As a result, only useful information is extracted
                  into three-dimensional space.
                  """),   
            html.H6('The result between the K-mean and DBSCAN clustering:'),
            html.P(
                  """
                  The calculation of silhouette coefficient can validate the similarity measure of
                  the cluster from a cohesion within the same cluster (using average distance in the cluster) and
                  separation between different clusters (using the nearest cluster distance).
                  The visualization can indicate that the K-Means algorithm produces the best discriminative clusters
                  with a silhouette score of 0.78, whereas DBCAN can identify more sub-clusters with an outlier but 
                  is less effective at clustering various densities of data.
                  """),                
        ])
    ],style={
             "width": "95%",
             "display": "inline-block",
             "margin-left": "20px",
             "margin-right": "20px",
             "margin-top": "20px",
             "margin-bottom": "10px",
             "text-indent": "35px",
             "text-align": "justify",
             "font-size":"14px"}
    ),
    html.Hr(style={'margin-bottom': '10px'}),
    #title box3
    dbc.Row([
       html.Div(children='Cluster Analysis',
              style={'background-color': 'background',
                     'margin': '20px'},
              className='text-center fw-bold'
              )
    ]),
    dbc.Row([ 
        html.Div([
            html.P(
                  """The pair plot can indicate the relationship between each cluster and its feature attributes. 
                  The gender is the most significant part in segmentation, followed by age, but annual income
                  has no significance.
                  """)
            ],style={
                    "width": "95%",
                    "display": "inline-block",
                    "margin-left": "20px",
                    "margin-right": "20px",
                    "margin-top": "10px",
                    "margin-bottom": "10px",
                    "text-indent": "35px",
                    "text-align": "justify",
                    "font-size":"14px"}
            )
    ]),
    #component2----------------------------------------------------------------
    dbc.Row([
        html.H5('Select Filter'),
        html.Hr(),
        dbc.Col([
            html.H6('Customer Gender:'),
            dcc.Dropdown(
                multi=True,
                id='ClusterSelect',
                options=[
                         {'label': x, 'value': x, 'disabled':False}
                         for x in df_seg_cluster['Cluster'].unique()
                ],
                value=['Cluster0', 'Cluster1', 'Cluster2', 'Cluster3'],
                style={'font-size': 15, 'padding-left': 10}
            ),
        ]),
        dbc.Col([
            #spending score slider
            html.H6('Spending Score:'),
            dcc.RangeSlider(
                df['Spending_Score'].min(), df['Spending_Score'].max(), 10, 
                value=[1, 99],
                id='Spending_ScoreSelect2'
            )
        ],style={'marginLeft': '20px','marginRight': '20px'})    
    ],style={'background-color': 'background',
              'paddingBottom': '20px'}),
    #--------------------------------------------------------------------------
    #graph4
    dbc.Row([ 
        dbc.Col([ # 4R/1C
              html.H5('Scatterplot Matrix of Cluster', className='text-center'),
              dcc.Graph(id='graph7',
                        figure=fig7)
        ]),
        dbc.Col([
              dbc.Row([ # 4R/2C/1R
                  html.H5('Cluster Ratio', className='text-center'),
                  dcc.Graph(id='graph9',
                            figure=fig9,
                            style={'height':300})
              ]),
              dbc.Row([ # 4R/2C/1R
                  html.H5('Box Plot of Cluster', className='text-center'),
                  dcc.Graph(id='graph8',
                            figure=fig8,
                            style={'height':300})
              ])
        ]),
    ]),
    #card
    html.Div([
        dbc.Row([
            dbc.Col(dbc.Card([
              dbc.CardHeader("Cluster0", className="card-title",
                             style={'background':'#0D0888',
                                    'color':'white'}),
              dbc.CardBody([
                  html.P("""Cluster 0 is made up of males aged 19 to 70, with a mean age of 49. 
                        They have spending scores under 80, with a mean of 29.20, which is the 
                        lowest spending group.).
                        """,className="card-text ",
                         )
                  ])
              ],style={'border-color':'#0D0888',
                       'font-size':'14px'})),

            dbc.Col(dbc.Card([
              dbc.CardHeader("Cluster 1", className="card-title",
                             style={'background':'#9D1A9F',
                                    'color':'white'}),
              dbc.CardBody([
                  html.P("""Cluster 1 is made up of females aged 20 to 68, with a mean age of 48.
                        This group has spending scores under 59, with a mean of 34.78.
                        (This is the second-lowest spending group).
                        """,className="card-text",)
                  ])
              ],style={'border-color':'#9D1A9F',
                       'font-size':'14px'})),

            dbc.Col(dbc.Card([
              dbc.CardHeader("Cluster 2", className="card-title",
                             style={'background':'#ED7953',
                                    'color':'white'}),
              dbc.CardBody([
                  html.P("""Cluster2 are females aged 18-40 years old with a mean age of 28 
                        years old and have the spending scores of more than 29 with a mean of 
                        67.68, which is the second-highest spending group.
                        """,className="card-text",)
                  ])
              ],style={'border-color':'#ED7953',
                       'font-size':'14px'})),
            dbc.Col(dbc.Card([
              dbc.CardHeader("Cluster 3", className="card-title",
                             style={'background':'#F5FB64',
                                    'color':'black'}),
              dbc.CardBody([
                  html.P("""Cluster3 are males aged 18–40 years old with a mean age of 28. 
                        This group has the spending scores from 39-97 with a mean of 71.67, 
                        which is the highest spending group.
                        """,className="card-text",)
                  ])
              ],style={'border-color':'#F5FB64',
                       'font-size':'14px'})),
        ],className="mb-4",
          style={'margin': '20px'})
    ])

], fluid=True,
   style={'padding':'2rem', 
          'margin':'1rem', 
          'boxShadow': '#e3e3e3 4px 4px 2px', 
          'border-radius': '10px', 
          'marginTop': '2rem',
          'backgroundColor': colors['main-background']})

#callback filter1-----------------------------------------------------------------------
@app.callback(
    [Output('graph3', 'figure'),
    Output('graph4', 'figure')],

    [Input('GenderSelect', 'value'),
    Input('AgeSelect', 'value'),
    Input('Annual_IncomeSelect', 'value'),
    Input('Spending_ScoreSelect', 'value')
    ]
)
def update_graph(GenderSelect, AgeSelect, Annual_IncomeSelect, Spending_ScoreSelect):
  #update gender
  #ilter_df = df[ (df['Gender'].isin(GenderSelect)) & (df['Age'].isin(AgeSelect)) ] 
  filter_df = df[df['Gender'].isin(GenderSelect)] 
  filter_df = filter_df.loc[(AgeSelect[0] <= filter_df['Age']) & (AgeSelect[1] >= filter_df['Age'])]
  filter_df = filter_df.loc[(Annual_IncomeSelect[0] <= filter_df['Annual_Income']) & (Annual_IncomeSelect[1] >= filter_df['Annual_Income'])]
  filter_df = filter_df.loc[(Spending_ScoreSelect[0] <= filter_df['Spending_Score']) & (Spending_ScoreSelect[1] >= filter_df['Spending_Score'])]

  filter_df_encode = filter_df.copy()
  filter_df_encode['Gender']= filter_df_encode['Gender'].map(gender)

  fig3 = ff.create_scatterplotmatrix(filter_df, diag='histogram', index='Gender',
                                  height=500, width=800, 
                                  colormap= dict(
                                      Female = '#FF9900',
                                      Male = '#1616A7'),
                                  colormap_type='cat',
                                  marker=dict(line_color='white', line_width=0.5)
                                  )
  
  filter_df_corr = filter_df_encode.corr().round(2)
  mask = np.triu(np.ones_like(filter_df_corr, dtype=bool))
  fig4 = go.Figure()
  fig4.add_trace(go.Heatmap( x=list(filter_df_corr.index.values),
                             y=list(filter_df_corr.columns.values),
                             z= filter_df_corr.mask(mask).to_numpy(),
                             colorscale=px.colors.diverging.RdBu
                             ))
  fig4.update_layout(yaxis_autorange='reversed')
  return fig3, fig4

#callback filter2-----------------------------------------------------------------------
@app.callback(
    [Output('graph7', 'figure'),
     Output('graph8', 'figure'),
     Output('graph9', 'figure')],

    [Input('ClusterSelect', 'value'),
    # Input('AgeSelect2', 'value'),
    # Input('Annual_IncomeSelect2', 'value'),
     Input('Spending_ScoreSelect2', 'value')
    ]
)

def update_graph(ClusterSelect,Spending_ScoreSelect2):
  #update cluster
  filter_df_seg = df_seg_cluster[df_seg_cluster['Cluster'].isin(ClusterSelect)] 
  filter_df_seg_gen = df_seg_gen[df_seg_gen['Cluster'].isin(ClusterSelect)] 
  filter_df_seg_feature = df_seg_feature[df_seg_feature['Cluster'].isin(ClusterSelect)] 

  #update spending score
  filter_df_seg = filter_df_seg.loc[(Spending_ScoreSelect2[0] <= filter_df_seg['Spending_Score']) & (Spending_ScoreSelect2[1] >= filter_df_seg['Spending_Score'])]
  filter_df_seg_gen = filter_df_seg_gen.loc[(Spending_ScoreSelect2[0] <= filter_df_seg_gen['Spending_Score']) & (Spending_ScoreSelect2[1] >= filter_df_seg_gen['Spending_Score'])]


  fig7 = ff.create_scatterplotmatrix(filter_df_seg, diag='histogram', index='Cluster',
                                     height=700, width=700,
                                     colormap= dict(
                                         Cluster0 = '#0D0888',
                                         Cluster1 = '#9D1A9F',
                                         Cluster2 = '#ED7953',
                                         Cluster3 = '#F5FB64'),
                                     colormap_type='cat',
                                     marker=dict(showscale=False, line_width=0.5),
                                     )
  
  fig8 = px.box(filter_df_seg_feature.sort_values(by=['Cluster'],ascending=False), x="features", y="value", color="Cluster",
             color_discrete_map={'Cluster0':'#0D0888',
                                 'Cluster1':'#9D1A9F',
                                 'Cluster2':'#ED7953',
                                 'Cluster3':'#F5FB64'}
            )

  fig9 = px.sunburst(filter_df_seg_gen, 
                  path=['Gender','Cluster'],
                  color='Cluster'
                  ,color_discrete_map={'(?)':'black','Cluster0':'#0D0888','Cluster1':'#9D1A9F','Cluster2':'#ED7953','Cluster3':'#F5FB64'}
                  )
  
  return fig7, fig8, fig9
  
if __name__ == '__main__':
    app.run_server(mode='external')
    #app.run_server(mode='inline', port=8030)
