$('#first_cat').on('change',function(){

    $.ajax({
        url: "/bar",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        data: {
            'selected': document.getElementById('first_cat').value

        },
        dataType:"json",
        success: function (data) {
            console.log(data)
            Plotly.newPlot('plot', data );
        }
    });
})

$('#stop').on('click',function(){

    $.ajax({
        url: "/stop",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        dataType:"json",
        success: function (data) {
            console.log("STOP")
        }
    });
    console.log("STOP!");
    socket.emit('disconnect', { data: 'disconnect' });
    //socket.disconnect();
})

$('#pause').on('click',function(){

    $.ajax({
        url: "/pause",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        dataType:"json",
        success: function (data) {
            console.log("PAUSE")
        }
    });
})

$('#resume').on('click',function(){
    console.log("resume")
    $.ajax({
        url: "/resume",
        type: "GET",
        contentType: 'application/json;charset=UTF-8',
        dataType:"json",
        success: function (data) {
            console.log("RESUME")
        }
    });
})

var socket;
$(document).ready(function(){
    //connect to the socket server.
    socket = io.connect('http://' + document.domain + ':' + location.port + '/general');
    var numbers_received = [];

    //receive details from server
    i = 0
    socket.on('plot_update', function(msg) {
        messages = eval(msg.log)
        for (i in messages)
            $("#log_table_body").append(messages[i])
        new_model0_errors = eval(msg.model0_errors)
        new_model_errors  = eval(msg.model_errors)
        x = eval(msg.x)
        graphs["data"][0].x.push(...x)//new_model0_errors.map((value, index) => x_max+(index+1)));
        graphs["data"][1].x = graphs["data"][0].x//new_model_errors.map((value, index) => x_max+(index+1)));
        graphs["data"][0].y.push(...new_model0_errors);
        graphs["data"][1].y.push(...new_model_errors);
        Plotly.redraw('plot');
    });
});
