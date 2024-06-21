import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import rclpy
from rclpy.node import Node
import re
import time
import subprocess
import threading
import logging
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='ROS 2 Node and Topic Explorer')
parser.add_argument('--hz_all', action='store_true', help='Enable Hz update for node page')
args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


# ===================================================================
# ======================= ros2 command thread =======================
# ===================================================================

# Global variables for thread management
thread_controls = {'hz': {}, 'echo': {}, 'param': {}}

def clear_threads(command_type, exclude_topic = None):
    """
    Clear separated threads for a specific command type.
    """
    global thread_controls
    topics_to_remove = []

    for topic, control in thread_controls[command_type].items():
        if topic == exclude_topic:
            continue
        control['stop'] = True
        if control['process']:
            control['process'].terminate()
        topics_to_remove.append(topic)

    # Remove all topics except the exclude_topic
    for topic in topics_to_remove:
        del thread_controls[command_type][topic]

def start_threads(command_type, topic_name, func):
    """
    Start separated threads for a specific command type and target function.
    """
    global thread_controls
    thread_controls[command_type][topic_name] = {'stop': False, 'process': None, 'messages': [], 'display': True }
    thread = threading.Thread(target=func, args=(
        topic_name, thread_controls[command_type][topic_name]), daemon=True)
    thread.start()

def run_ros2_param(node_name, param_control_dict, loop=False):
    """
    Run ros2 echo command in a separate thread and put its output in a message.
    """

    # Not necessary if you run source before running this app
    # cmd = "source /home/user/hogehoge/setup.bash && ros2 param dump" + node_name"
    cmd = "ros2 param dump " + node_name
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True, executable='/bin/bash', text=True)

    max_line = 1000

    while True:
        output = process.stdout.readline()
        if output and param_control_dict['display']:
            msg = output.rstrip('\n')
            param_control_dict['messages'].append(msg)
            if len(param_control_dict['messages']) > max_line:
                param_control_dict['messages'].pop(0)
            # logging.info(f"message: {msg}")  # show message on terminal
        else:
            time.sleep(0.1)

        if param_control_dict['stop'] is True:
            logging.info(f"{node_name}: END thank you!")
            break


def run_ros2_echo(topic_name, echo_control_dict, loop=False):
    """
    Run ros2 echo command in a separate thread and put its output in a message.
    """

    # Not necessary if you run source before running this app
    # cmd = "source /home/user/hogehoge/setup.bash && ros2 topic echo " + topic_name + " --once"
    cmd_option = "" if loop else " --once"
    cmd = "ros2 topic echo " + topic_name + cmd_option
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True, executable='/bin/bash', text=True)

    max_line = 1000

    while True:
        output = process.stdout.readline()
        if output and echo_control_dict['display']:
            msg = output.strip()
            echo_control_dict['messages'].append(msg)
            if len(echo_control_dict['messages']) > max_line:
                echo_control_dict['messages'].pop(0)
            # logging.info(f"message: {msg}")  # show message on terminal
        else:
            time.sleep(0.1)

        if echo_control_dict['stop'] is True:
            logging.info(f"{topic_name}: END thank you!")
            break


def run_ros2_hz(topic_name, control_dict):
    """
    Run ros2 hz command in a separate thread and put its output in a message.
    """

    # Not necessary if you run source before running this app
    # cmd = "source /home/user/hogehoge/setup.bash && ros2 topic hz " + topic_name  
    cmd = "ros2 topic hz " + topic_name
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True, executable='/bin/bash', text=True)

    max_line = 100

    while True:
        output = process.stdout.readline()

        if output and control_dict['display']:
            msg = output.strip()
            control_dict['messages'].append(msg)
            if len(control_dict['messages']) > max_line:
                control_dict['messages'].pop(0)
            # logging.info(f"message: {msg}")  # show message on terminal
        else:
            time.sleep(0.1)

        if control_dict['stop'] is True:
            logging.info(f"{topic_name}: END thank you!")
            break

# ===================================================================
# ======================= for ros2 interface ========================
# ===================================================================
class ROS2InfoNode(Node):
    def __init__(self):
        super().__init__('ros2_info_node')

    def get_all_topics(self):
        return self.get_topic_names_and_types()

    def get_all_nodes(self):
        combined_names = []
        for name, namespace in self.get_node_names_and_namespaces():
            if name.startswith("_"):
                continue  # skip for _ros2cli, etc
            combined_name = f"{namespace}/{name}" if namespace != '/' else f"/{name}"
            combined_names.append(combined_name)
        return combined_names

    def get_topics_by_node(self, node_name):
        name, namespace = self.extract_name_and_namespace(node_name)
        pubs = self.get_publisher_names_and_types_by_node(name, namespace)
        subs = self.get_subscriber_names_and_types_by_node(name, namespace)
        return {'publishers': pubs, 'subscribers': subs}

    def get_nodes_by_topic(self, topic_name):
        pub_nodes = self.get_publishers_info_by_topic(topic_name)
        sub_nodes = self.get_subscriptions_info_by_topic(topic_name)
        publishers, subscribers = [], []
        for pub in pub_nodes:
            if pub.node_name.startswith("_"):
                continue  # skip for _ros2cli, etc
            publisher_name = pub.node_namespace + "/" + \
                pub.node_name if pub.node_namespace != '/' else '/' + pub.node_name
            publishers.append(publisher_name)
        for sub in sub_nodes:
            if sub.node_name.startswith("_"):
                continue  # skip for _ros2cli, etc
            subscriber_name = sub.node_namespace + "/" + \
                sub.node_name if sub.node_namespace != '/' else '/' + sub.node_name
            subscribers.append(subscriber_name)
        return {'publishers': publishers, 'subscribers': subscribers}

    @staticmethod
    def extract_name_and_namespace(full_name):
        parts = full_name.split('/')
        return parts[-1], '/'.join(parts[:-1])


# Initialize ROS2
rclpy.init()
ros2_node = ROS2InfoNode()

# Initialize Dash app with external stylesheet
external_stylesheets = ['https://fonts.googleapis.com/css2?family=Lato:wght@300;700&display=swap']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'ROS 2 Explorer'

# wait for ros connection, otherwise ros information is not available
time.sleep(1)


default_div_style = {'maxHeight': '25em', 'overflowY': 'scroll', 'border': '1px solid black', 'marginBottom': '2em'}
default_input_style = {'marginBottom': '10px', 'height': '2.0em', 'width': '50em'}

pubsub_info_style = default_div_style.copy()
pubsub_info_style['minHeight'] = '30em'
pubsub_info_style['maxHeight'] = '40em'
pubsub_info_style['overflowY'] = 'hidden'

app.layout = html.Div(style={'padding': '20px', 'marginBottom': '10px', 'font-family': 'Lato', 'fontWeight': '300'}, children=[
    
    # Title
    html.H1("ROS 2 Explorer"),

    # Update trigger
    dcc.Location(id='url', refresh=False),

    # Main information
    html.Div(id='main-info', style=pubsub_info_style),

    # ros topic list
    html.H2("ROS 2 Topic List", style={'marginTop': '80px'}),
    dcc.Input(id='topic-filter', type='text', placeholder='Filter topics by regex pattern', style=default_input_style),
    html.Div(id='topic-list', style=default_div_style),
    
    # ros node list
    html.H2("ROS 2 Node List", style={'marginTop': '80px'}),
    dcc.Input(id='node-filter', type='text', placeholder='Filter nodes by regex pattern', style=default_input_style),
    html.Div(id='node-list', style=default_div_style),

    # timer update for node/topic list
    dcc.Interval(id='interval-update-node-topic-list', interval=10*1000, n_intervals=0),

    html.Div(id='dummy-output', style={'display': 'none'}),  # This Div is hidden and used only for callback purposes
])

# ===================================================================
# ========= For main (ros2 node /ros2 topic) info division ==========
# ===================================================================
@app.callback(
    Output('main-info', 'children'),
    Input('url', 'pathname')
)
def update_main_info(pathname):

    """
    Update topic information based on the URL pathname.

    Args:
    pathname (str): The URL pathname used to extract the topic or node name.

    Returns:
    html.Div: A Div containing the ROS 2 topic or node information.
    """

    def extract_name(name_key):
        """Extract the name from the pathname using a specified key."""
        return pathname.split(f'{name_key}')[-1]


    global thread_controls

    if 'topic_name=' in pathname:
        topic_name = extract_name('topic_name=')

        # Stop the existing thread for the old topic and start a new one for the new topic
        clear_threads('hz')

        clear_threads('hz', exclude_topic=topic_name)
        if topic_name not in thread_controls['hz']:
            start_threads('hz', topic_name, run_ros2_hz)

        clear_threads('echo')
        if topic_name not in thread_controls['echo']:
            start_threads('echo', topic_name, run_ros2_echo)

        else:
            logging.info(
                f"update_topic_info: topic_name {topic_name} is in thread_control.")
 
        return generate_main_topic_info_div(topic_name)
    
    elif 'node_name=' in pathname:
        node_name = extract_name('node_name=')

        # start topic hz thread for all pubsub topics
        topics = ros2_node.get_topics_by_node(node_name)
        
        clear_threads('hz')
        if args.hz_all:
            for topic in topics['publishers'] + topics['subscribers']:
                topic_name = topic[0]
                if topic_name not in thread_controls['hz']:
                    start_threads('hz', topic_name, run_ros2_hz)
        
        clear_threads('param')
        if node_name not in thread_controls['param']:
            start_threads('param', node_name, run_ros2_param)

        return generate_main_node_info_div(node_name, topics)

    else:
        logging.info(f"UNKNOWN mode: URL = {pathname}")

    return html.H2("Select a topic or node to see the information.")



# ===================================================================
# ======================== Topic list update ========================
# ===================================================================
def get_topic_list(pattern=None):
    topics = ros2_node.get_all_topics()
    if pattern:
        regex_pattern = re.compile(pattern)
        topics = filter(lambda topic: regex_pattern.search(topic[0]), topics)
    return [{'topic_name_type': f"{topic} [{topic_type[0]}]", 'name': topic} for topic, topic_type in topics]

@app.callback(
    Output('topic-list', 'children'),
    Input('interval-update-node-topic-list', 'n_intervals'),
    Input('topic-filter', 'value')
)
def update_topic_list(n_intervals, topic_pattern):
    topic_options = get_topic_list(topic_pattern)
    topic_list = html.Ul([
        html.Li([
            html.A(topic_option['topic_name_type'],
                   href=f"/topic_name={topic_option['name']}")
        ]) for topic_option in topic_options])
    return topic_list

# ===================================================================
# ========================= Node list update ========================
# ===================================================================
def get_node_list(pattern=None):
    nodes = ros2_node.get_all_nodes()
    if pattern:
        regex_pattern = re.compile(pattern)
        nodes = filter(lambda node: regex_pattern.search(node), nodes)
    return sorted([f"{node}" for node in nodes])

@app.callback(
    Output('node-list', 'children'),
    Input('interval-update-node-topic-list', 'n_intervals'),
    Input('node-filter', 'value')
)
def update_node_list(n_intervals, node_pattern):
    node_names = get_node_list(node_pattern)
    node_list = html.Ul([
        html.Li(html.A(node, href=f"/node_name={node}")) for node in node_names
    ])
    return node_list


# ===================================================================
# ================== For ros2 topic info division ===================
# ===================================================================
def generate_main_topic_info_div(topic_name):
        nodes = ros2_node.get_nodes_by_topic(topic_name)

        def create_node_links(node_list):
            """Create HTML links for each node in the node list."""
            sorted_node_list = sorted(node_list)
            return [html.A(f"{node}", href=f"/node_name={node}") for node in sorted_node_list]

        publishers, subscribers = create_node_links(nodes['publishers']), create_node_links(nodes['subscribers'])

        pre_style = {'word-wrap': 'break-word', 'white-space': 'pre-wrap', 'margin-left': '1rem', 'margin-right': '1rem'}

        return_item = html.Div([
            html.Div([
                html.H3(f"ros2 topic info {topic_name}", style={'margin-left': '1em'}),
                html.Div([
                    html.H4("Publishers:", style={'margin-left': '2em'}),
                    html.Ul([html.Li(publisher) for publisher in publishers]),
                    html.H4("Subscribers:", style={'margin-left': '2em'}),
                    html.Ul([html.Li(subscriber) for subscriber in subscribers]),
                ], style={'overflowY': 'auto', 'maxHeight': '30em', 'margin': '1em'}),
                dcc.Interval(id='update-interval-topic-page-hz', interval=200, n_intervals=0),
                dcc.Interval(id='update-interval-topic-page-echo', interval=200, n_intervals=0),
            ], style={'width': '50%', 'float': 'left'}),
            html.Div([
                html.H3("ros2 topic hz", style={'margin-left': '1em'}),
                html.Div([
                    html.Pre(id='hz-output', children='Loading...', style=pre_style),
                ], style={'overflowY': 'auto', 'maxHeight': '30em', 'margin': '1em'}),
            ], style={'width': '25%', 'float': 'left'}),
            html.Div([
                # Header and buttons aligned to the left in a flex container
                html.Div([
                    html.H3("ros2 topic echo --once", style={'margin-left': '1em'}),
                    html.Button('once', id='echo-reset-button', n_clicks=0, style={'marginLeft': '2em', 'marginRight': '0.5em', 'height': '1.5em'}),
                    html.Button('loop', id='loop-button', n_clicks=0, style={'height': '1.5em'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'justify-content': 'flex-start'}),

                # Scrollable div for echo output
                html.Div([
                    html.Pre(id='echo-output', children='Loading...', style=pre_style)
                ], style={'overflowY': 'auto', 'maxHeight': '30em', 'marginRight': '1em', 'marginLeft': '1em', 'marginBottom': '1em'}),
            ], style={'width': '25%', 'float': 'left'})
        ])
        return return_item

def update_thread_output(thread_control, pathname):
    if 'topic_name=' in pathname:
        topic_name = pathname.split('topic_name=')[-1]
        if topic_name in thread_control:
            control_dict = thread_control[topic_name]
            if control_dict['messages']:
                return "\n".join(control_dict['messages'])
            else:
                return "No output received..."
        else:
            logging.info(
                f"topic_name = {topic_name} is NOT in thread control. thread_control = {thread_control}")

    if 'node_name=' in pathname:
        node_name = pathname.split('node_name=')[-1]
        if node_name in thread_control:
            control_dict = thread_control[node_name]
            if control_dict['messages']:
                return "\n".join(control_dict['messages'])
            else:
                return "No output received..."
        else:
            logging.info(
                f"node_name = {node_name} is NOT in thread control. thread_control = {thread_control}")

    return 'None'

# --- hz update in topic info page ---
@app.callback(
    Output('hz-output', 'children'),
    [Input('update-interval-topic-page-hz', 'n_intervals')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_hz_output(n_intervals, pathname):
    global thread_controls
    return update_thread_output(thread_controls['hz'], pathname)


# --- hz update in topic info page ---
@app.callback(
    Output('echo-output', 'children'),
    [Input('update-interval-topic-page-echo', 'n_intervals')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_hz_output(n_intervals, pathname):
    global thread_controls
    return update_thread_output(thread_controls['echo'], pathname)


# --- Callback for the "once" button in topic info page ---
@app.callback(
    Output('dummy-output', 'children'),  # Using a dummy output as no actual output is needed
    Input('echo-reset-button', 'n_clicks'),
    [dash.dependencies.State('url', 'pathname')]
)
def reset_echo(n_clicks, pathname):
    global thread_controls

    if 'topic_name=' in pathname and n_clicks > 0:  # Check if the button was clicked
        topic_name = pathname.split('topic_name=')[-1]

        # Clear the echo threads and start a new one
        clear_threads('echo')
        start_threads('echo', topic_name, run_ros2_echo)

    # Return is required for a callback, but in this case, it does nothing
    return None

# --- Callback for the "loop" button in topic info page ---
@app.callback(
    Output('loop-button', 'style'),
    Input('loop-button', 'n_clicks'),
    [dash.dependencies.State('url', 'pathname')]
)
def toggle_loop(n_clicks, pathname):
    global thread_controls

    # Toggle state: odd clicks mean looping, even clicks mean not looping. 0 means initial time.
    clicked = n_clicks % 2 == 1
    if 'topic_name=' in pathname:
        topic_name = pathname.split('topic_name=')[-1]

        if clicked:
            clear_threads('echo')
            start_threads('echo', topic_name, lambda tn, ctrl_dict: run_ros2_echo(tn, ctrl_dict, loop=True))
        elif n_clicks != 0:  # if this is not the initial time
            thread_controls['echo'][topic_name]['display'] = False

    # Change color based on whether it's an odd or even click
    return {'backgroundColor': 'darkgray'} if clicked else {}


# ===================================================================
# =================== For ros2 node info division ===================
# ===================================================================
def generate_main_node_info_div(node_name, topics):

        publishers = [html.A(f"{topic[0]} [{topic[1][0]}]", href=f"/topic_name={topic[0]}")
                      for topic in topics['publishers']]
        subscribers = [html.A(f"{topic[0]} [{topic[1][0]}]", href=f"/topic_name={topic[0]}")
                       for topic in topics['subscribers']]
        
        pre_style = {'word-wrap': 'break-word', 'white-space': 'pre-wrap', 'margin-left': '1rem', 'margin-right': '1rem'}

        children = [
            html.Div([
            html.H3(f"ros2 node info {node_name}", style={'margin-left': '1em'}),
            html.Div([
                html.H4("Publishing Topics:", style={'margin-left': '2em'}),
                html.Ul([html.Li(publisher) for publisher in publishers], id='publishers-list'),
                html.H4("Subscribing Topics:", style={'margin-left': '2em'}),
                html.Ul([html.Li(subscriber) for subscriber in subscribers], id='subscribers-list'),
            ], style={'overflowY': 'auto', 'maxHeight': '30em', 'margin': '1em'}),
            ], style={'width': '70%', 'float': 'left'}),
            html.Div([
                html.H3("ros2 param dump", style={'margin-left': '1em'}),
                html.Div([
                    html.Pre(id='param-output', children='Loading...', style=pre_style),
                ], style={'overflowY': 'auto', 'maxHeight': '30em', 'margin': '1em'}),
            ], style={'width': '30%', 'float': 'left'}),
            dcc.Interval(id='update-interval-node-page-param', interval=200, n_intervals=0),
        ]

        # Conditionally add Interval component
        if args.hz_all:
            children.append(dcc.Interval(id='interval-update-node-page-pubsub-hz', interval=200, n_intervals=0))

        return html.Div(children)

# --- param update in node info page ---
@app.callback(
    Output('param-output', 'children'),
    [Input('update-interval-node-page-param', 'n_intervals')],
    [dash.dependencies.State('url', 'pathname')]
)
def update_hz_output(n_intervals, pathname):
    global thread_controls
    return update_thread_output(thread_controls['param'], pathname)


def extract_rate_from_hz_message(topic_name):
    global thread_controls

    hz_control = thread_controls.get('hz', {})
    topic_control = hz_control.get(topic_name, {})
    hz_messages = topic_control.get('messages', [])

    if hz_messages:
        for message in reversed(hz_messages):  # Iterate in reverse to start from the most recent message
            match = re.search(r"average rate: ([0-9.]+)", message)
            if match:
                return float(match.group(1))
    return '-'

# --- Update node pubsub hz ---
@app.callback(
    [Output('publishers-list', 'children'),
     Output('subscribers-list', 'children')],
    Input('interval-update-node-page-pubsub-hz', 'n_intervals'),
    [dash.dependencies.State('url', 'pathname')]
)
def update_node_pubsub_hz(n_intervals, pathname):
    def extract_name(name_key):
        """Extract the name from the pathname using a specified key."""
        return pathname.split(f'{name_key}')[-1]
    

    logging.info(f"thread_controls len = {len(thread_controls['echo'])}, {len(thread_controls['hz'])}")

    if 'node_name=' in pathname:
        node_name = extract_name('node_name=')
        topics = ros2_node.get_topics_by_node(node_name)

        updated_publishers = [
            html.Li(html.A(f"{topic[0]} [{topic[1][0]}] ({extract_rate_from_hz_message(topic[0])} Hz)",
                           href=f"/topic_name={topic[0]}"))
            for topic in topics['publishers']
        ]
        updated_subscribers = [
            html.Li(html.A(f"{topic[0]} [{topic[1][0]}] ({extract_rate_from_hz_message(topic[0])} Hz)",
                           href=f"/topic_name={topic[0]}"))
            for topic in topics['subscribers']
        ]

        return updated_publishers, updated_subscribers

    return [], []  # Return empty lists if pathname does not include 'node_name='

def main():
    app.run_server(debug=False)

if __name__ == '__main__':
    main()
