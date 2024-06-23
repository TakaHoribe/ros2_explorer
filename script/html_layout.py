from dash import html, dcc

def generate_main_layout():
    default_div_style = {'maxHeight': '25em', 'overflowY': 'scroll', 'border': '1px solid black', 'marginBottom': '2em'}
    default_input_style = {'marginBottom': '10px', 'height': '2.0em', 'width': '50em'}

    pubsub_info_style = default_div_style.copy()
    pubsub_info_style['minHeight'] = '30em'
    pubsub_info_style['maxHeight'] = '40em'
    pubsub_info_style['overflowY'] = 'hidden'

    return html.Div(style={'padding': '20px', 'marginBottom': '10px', 'font-family': 'Lato', 'fontWeight': '300'}, children=[

        # Title
        html.H1("ROS 2 Explorer"),

        # Update trigger
        dcc.Location(id='url', refresh=False),

        # Main information
        html.Div(id='main-info', style=pubsub_info_style),

        # ros topic list
        html.H2("ROS 2 Topic List", style={'marginTop': '80px'}),
        dcc.Input(id='topic-filter', type='text',
                placeholder='Filter topics by regex pattern', style=default_input_style),
        html.Div(id='topic-list', style=default_div_style),

        # ros node list
        html.H2("ROS 2 Node List", style={'marginTop': '80px'}),
        dcc.Input(id='node-filter', type='text',
                placeholder='Filter nodes by regex pattern', style=default_input_style),
        html.Div(id='node-list', style=default_div_style),

        # timer update for node/topic list
        dcc.Interval(id='interval-update-node-topic-list',
                    interval=10*1000, n_intervals=0),

        # This Div is hidden and used only for callback purposes
        html.Div(id='dummy-output', style={'display': 'none'}),
    ])

def generate_pubsub_layout(topic_name, publishers, subscribers):

    pre_style = {'word-wrap': 'break-word', 'white-space': 'pre-wrap', 'margin-left': '1rem', 'margin-right': '1rem'}

    return html.Div([
        html.Div([
            html.H3(f"ros2 topic info {topic_name}",
                    style={'margin-left': '1em'}),
            html.Div([
                html.H4("Publishers:", style={'margin-left': '2em'}),
                html.Ul([html.Li(publisher) for publisher in publishers]),
                html.H4("Subscribers:", style={'margin-left': '2em'}),
                html.Ul([html.Li(subscriber) for subscriber in subscribers]),
            ], style={'overflowY': 'auto', 'maxHeight': '30em', 'margin': '1em'}),
            dcc.Interval(id='update-interval-topic-page-hz',
                         interval=200, n_intervals=0),
            dcc.Interval(id='update-interval-topic-page-echo',
                         interval=200, n_intervals=0),
        ], style={'width': '50%', 'float': 'left'}),
        html.Div([
            html.H3("ros2 topic hz", style={'margin-left': '1em'}),
            html.Div([
                html.Pre(id='hz-output', children='Loading...',
                         style=pre_style),
            ], style={'overflowY': 'auto', 'maxHeight': '30em', 'margin': '1em'}),
        ], style={'width': '25%', 'float': 'left'}),
        html.Div([
            # Header and buttons aligned to the left in a flex container
            html.Div([
                html.H3("ros2 topic echo --once",
                        style={'margin-left': '1em'}),
                html.Button('once', id='echo-reset-button', n_clicks=0,
                            style={'marginLeft': '2em', 'marginRight': '0.5em', 'height': '1.5em'}),
                html.Button('loop', id='loop-button', n_clicks=0,
                            style={'height': '1.5em'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'justify-content': 'flex-start'}),

            # Scrollable div for echo output
            html.Div([
                html.Pre(id='echo-output',
                         children='Loading...', style=pre_style)
            ], style={'overflowY': 'auto', 'maxHeight': '30em', 'marginRight': '1em', 'marginLeft': '1em', 'marginBottom': '1em'}),
        ], style={'width': '25%', 'float': 'left'})
    ])
