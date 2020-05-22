import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import time
import sys
from random    import random
from time      import sleep
from threading import Thread, Event
from SPC       import GeneratorFromFile, SPC
from flask          import Flask, request, render_template, url_for, copy_current_request_context
from flask_socketio import SocketIO, emit
from skmultiflow.data.waveform_generator import WaveformGenerator
from skmultiflow.drift_detection         import DDM
from skmultiflow.bayes                   import NaiveBayes
from skmultiflow.rules                   import VFDR
from skmultiflow.trees                   import HoeffdingTree
from skmultiflow.meta                    import AdaptiveRandomForest
from skmultiflow.data                    import ConceptDriftStream


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
BASE_DIR = "../../../datasets/"

socketio = SocketIO(app, async_mode=None, logger=False, engineio_logger=False)

@app.route('/')
def index():
    print("index")
    return render_template('index.html')

@app.route('/simulation')
def simulation():
    global thread
    global stream_stop_event
    global stream_pause_event
    thread = Thread()
    stream_stop_event = Event()
    stream_stop_event.set()
    stream_pause_event = Event()
    dataset = request.args.get('dataset')+".data"
    print("DATASET:", dataset)
    model_name = request.args.get('model')
    if model_name == "NaiveBayes":
        model = NaiveBayes()
    elif model_name == "VFDR":
        model = VFDR(ordered_rules=False, rule_prediction="weighted_sum", drift_detector=None)
    else:
        model = HoeffdingTree()
    freq    = request.args.get('freq')
    alpha   = request.args.get('alpha')
    beta    = request.args.get('beta')
    buffer  = True if request.args.get('buffer')=="on" else False
    xmax    = pd.read_csv(BASE_DIR+dataset).shape[0]+1
    thread  = socketio.start_background_task(spc_method, dataset, model, int(alpha), int(beta), buffer, int(freq))
    plot    = create_plot(model_name, xmax)
    return render_template('simulation.html', plot=plot)

def create_plot(model_name, xmax):
    #N = 40
    #x = []#np.linspace(0, 1, N)
    #y = []#np.random.randn(N)
    #df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe
    data = [go.Scatter( x=[], y=[], name=model_name + " without Change Detection"),
            go.Scatter( x=[], y=[], name=model_name + " with Change Detection")]
    graphJSON = {"data":json.loads(json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder))}
    graphJSON["layout"] = {"title": {"text":"Statistical Process Control with "+model_name},
                           "xaxis":{"title":{"text":"t"}, "range":[0, xmax]},
                           "yaxis":{"title":{"text":"Error Rate"}, "range":[0, 1]}}
    graphJSON = json.dumps(graphJSON, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/resume')
def resume():
    stream_pause_event.clear()
    stream_stop_event.clear()
    return "True"

@app.route('/pause')
def pause():
    stream_pause_event.set()
    return "True"

@app.route('/stop')
def stop():
    #stream_pause_event.set()
    stream_stop_event.set()
    return "True"

def spc_method(dataset, model, alpha, beta, with_buffer, freq):
    freq = int(freq/10)
    spc = SPC(model, GeneratorFromFile(BASE_DIR+dataset), alpha, beta, with_buffer)
    spc.start()
    stream = WaveformGenerator()
    stream.prepare_for_use()
    pause = False
    global xmax
    xmax = 0
    while not spc.has_finished():
        time.sleep(1)
    log = spc.get_messages()
    keys = log.keys()
    def build_row(k, msg):
        bg = "danger"
        if msg == "Warning": bg = "warning"
        elif msg == "In Control": bg = "primary"
        return "<tr class='bg-"+bg+"'><td style='width:20%' scope='row'>"+str(k)+"</td><td style='width:80%'>"+msg+"</td></tr>"
    while stream_stop_event.isSet(): socketio.sleep(0.1)
    while not stream_stop_event.isSet():
        while not stream_pause_event.isSet():
            errors0, errors = spc.get_next_errors(freq)
            if errors == []: stream_pause_event.set()
            n_errors = len(errors)
            filtered_keys = [k for k in keys if xmax < k < xmax+n_errors]
            messages = [build_row(k, log[k]) for k in filtered_keys]
            socketio.emit('plot_update', {'model0_errors': errors0, "x": list(range(xmax, xmax+n_errors)), 'model_errors': errors, 'log': messages}, namespace='/general')
            xmax += n_errors
            socketio.sleep(0.1)

@socketio.on('connect', namespace='/general')
def test_connect():
    global thread
    print('Client connected')


@socketio.on('disconnect', namespace='/general')
def test_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app)

