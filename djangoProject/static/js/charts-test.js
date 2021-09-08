
$(document).ready(function () {

    'use strict';

    Chart.defaults.global.defaultFontColor = '#75787c';
    var legendState = true;
    if ($(window).outerWidth() < 576) {
        legendState = false;
    }

    var LINECHART1 = $('#lineChartCustom2');
    var datalabel = timing ;
    var datax = motionx;
    var datay = motiony;
    var xmax = xmax;
    var ymin = ymin;

    var step = (xmax-ymin)/20;


    var myLineChart = new Chart(LINECHART1, {
        type: 'line',
        options: {
            scales: {
                xAxes: [{
                    display: true,
                    gridLines: {
                        display: true
                    }
                }],
                yAxes: [{
                    ticks: {
                        max: xmax,
                        min: ymin,
                        stepSize: step
                    },
                    display: true,
                    gridLines: {
                        display: true
                    }
                }]
            },
            legend: {
                display: legendState
            }
        },
        data: {
            labels: datalabel,
            datasets: [
                {
                    label: "X-coordinate",
                    fill: true,
                    lineTension: 0.3,
                    backgroundColor: "transparent",
                    borderColor: '#EF8C99',
                    pointBorderColor: '#EF8C99',
                    pointHoverBackgroundColor: '#EF8C99',
                    borderCapStyle: 'butt',
                    borderDash: [],
                    borderDashOffset: 0.0,
                    borderJoinStyle: 'miter',
                    borderWidth: 2,
                    pointBackgroundColor: "#EF8C99",
                    pointBorderWidth: 2,
                    pointHoverRadius: 4,
                    pointHoverBorderColor: "#fff",
                    pointHoverBorderWidth: 0,
                    pointRadius: 2,
                    pointHitRadius: 0,
                    data: datax,
                    spanGaps: false
                },
                {
                    label: "Y-coordinate",
                    fill: true,
                    lineTension: 0.3,
                    backgroundColor: "transparent",
                    borderColor: '#864DD9',
                    pointBorderColor: '#864DD9',
                    pointHoverBackgroundColor: '#864DD9',
                    borderCapStyle: 'butt',
                    borderDash: [],
                    borderDashOffset: 0.0,
                    borderJoinStyle: 'miter',
                    borderWidth: 2,
                    pointBackgroundColor: "#864DD9",
                    pointBorderWidth: 2,
                    pointHoverRadius: 4,
                    pointHoverBorderColor: "#fff",
                    pointHoverBorderWidth: 0,
                    pointRadius: 2,
                    pointHitRadius: 0,
                    data: datay,
                    spanGaps: false
                }
            ]
        }
    });

    var BARCHART1 = $('#barChartCustom1');
    var linkNum1 = linkNum;
    var linklength1 = linklength;
    var maxlinklength = maxlength;
    var barChartHome1 = new Chart(BARCHART1, {
        type: 'bar',
        options:
        {
            scales:
            {
                xAxes: [{
                    display: true,
                    barPercentage: 0.4
                }],
                yAxes: [{
                    ticks: {
                        max: maxlinklength,
                        min: 0
                    },
                    display: false
                }],
            },
            legend: {
                display: false
            }
        },
        data: {
            labels: linkNum1,
            datasets: [
                {
                    label: "LinkLength",
                    backgroundColor: [
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99'
                    ],
                    borderColor: [
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99',
                        '#EF8C99'
                    ],
                    borderWidth: 0.4,
                    data: linklength1
                }
            ]
        }
    });



    var BARCHART2 = $('#barChartCustom2');
    var dataerror = error_list;
    var mechanical = mechanical_num;
    var barChartHome2 = new Chart(BARCHART2, {
        type: 'bar',
        options:
        {
            scales:
            {
                xAxes: [{
                    display: true,
                    barPercentage: 0.5
                }],
                yAxes: [{
                    ticks: {
                        max: 500,
                        min: 0,
                        stepSize:50
                    },
                    display: true
                }],
            },
            legend: {
                display: false
            }
        },
        data: {
            labels: mechanical,
            datasets: [
                {
                    label: "FittingError",
                    backgroundColor: [
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9'
                    ],
                    borderColor: [
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9',
                        '#CF53F9'
                    ],
                    borderWidth: 0.2,
                    data: dataerror
                }
            ]
        }
    });

var LINECHART3 = $('#lineChartCustom3');
    var time = timing ;
    var vxy = endVelocity;
    var axy = endAcceleration;

    var Vmax = Vmax;
    var Vmin = Vmin;

    var vstep = (Vmax-Vmin)/100;

    var LineChar3 = new Chart(LINECHART3, {
        type: 'line',
        options: {
            scales: {
                xAxes: [{
                    display: true,
                    gridLines: {
                        display: true
                    }
                }],
                yAxes: [{
                    ticks: {
                        max: Vmax,
                        min: Vmin,
                        stepSize: vstep
                    },
                    display: true,
                    gridLines: {
                        display: true
                    }
                }]
            },
            legend: {
                display: legendState
            }
        },
        data: {
            labels: time,
            datasets: [
                {
                    label: "endPoint-Velocity",
                    fill: true,
                    lineTension: 0.3,
                    backgroundColor: "transparent",
                    borderColor: '#CF53F9',
                    pointBorderColor: '#CF53F9',
                    pointHoverBackgroundColor: '#CF53F9',
                    borderCapStyle: 'butt',
                    borderDash: [],
                    borderDashOffset: 0.0,
                    borderJoinStyle: 'miter',
                    borderWidth: 2,
                    pointBackgroundColor: "#CF53F9",
                    pointBorderWidth: 2,
                    pointHoverRadius: 4,
                    pointHoverBorderColor: "#fff",
                    pointHoverBorderWidth: 0,
                    pointRadius: 2,
                    pointHitRadius: 0,
                    data: vxy,
                    spanGaps: false
                },
                {
                    label: "endPoint-Acceleration",
                    fill: true,
                    lineTension: 0.3,
                    backgroundColor: "transparent",
                    borderColor: '#117a8b',
                    pointBorderColor: '#117a8b',
                    pointHoverBackgroundColor: '#117a8b',
                    borderCapStyle: 'butt',
                    borderDash: [],
                    borderDashOffset: 0.0,
                    borderJoinStyle: 'miter',
                    borderWidth: 2,
                    pointBackgroundColor: "#117a8b",
                    pointBorderWidth: 2,
                    pointHoverRadius: 4,
                    pointHoverBorderColor: "#fff",
                    pointHoverBorderWidth: 0,
                    pointRadius: 2,
                    pointHitRadius: 0,
                    data: axy,
                    spanGaps: false
                }
            ]
        },
    });

});