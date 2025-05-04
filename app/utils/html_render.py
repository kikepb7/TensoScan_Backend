import os
from datetime import datetime
from pygments.lexers import go

def render_measurements_html(measurements: list) -> str:
    fechas = []
    sistolica = []
    diastolica = []
    pulsaciones = []
    table_rows_html = ""

    for measure in measurements:
        fecha = measure["timestamp"].strftime('%d-%m-%Y') if hasattr(measure["timestamp"], 'strftime') else str(measure["timestamp"])
        high_pressure = measure["result"]["high_pressure"]
        low_pressure = measure["result"]["low_pressure"]
        pulse = measure["result"]["pulse"]

        fechas.append(fecha)
        sistolica.append(high_pressure)
        diastolica.append(low_pressure)
        pulsaciones.append(pulse)

        table_rows_html += f"""
            <tr>
                <td>{high_pressure}</td>
                <td>{low_pressure}</td>
                <td>{pulse}</td>
                <td>{fecha}</td>
            </tr>
        """

    return f"""
    <html>
    <head>
        <title>Informe de Mediciones</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 20px;
                background-color: #f4f7fc;
            }}
            h1, h2 {{
                text-align: center;
                color: #333;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background-color: #fff;
                border-radius: 10px;
            }}
            th, td {{
                padding: 12px;
                text-align: center;
                font-size: 14px;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .table-container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .btn-download {{
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Informe de Mediciones</h1>
        <h2>Fecha de generación: {datetime.now().strftime('%d-%m-%Y')}</h2>

        <div style="text-align: center; margin-top: 30px;">
            <a href="/ocr/user/measurements/pdf" class="btn-download" target="_blank">
                Descargar PDF
            </a>
        </div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Sistólica</th>
                        <th>Diastólica</th>
                        <th>Pulso</th>
                        <th>Fecha</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
        </div>

        <div id="grafico" style="margin-top: 50px;"></div>

        <script>
            var fechas = {fechas};
            var sistolica = {sistolica};
            var diastolica = {diastolica};
            var pulsaciones = {pulsaciones};

            var trace1 = {{
                x: fechas,
                y: sistolica,
                type: 'bar',
                name: 'Presión Sistólica',
                marker: {{ color: 'rgb(255, 99, 132)' }}
            }};

            var trace2 = {{
                x: fechas,
                y: diastolica,
                type: 'bar',
                name: 'Presión Diastólica',
                marker: {{ color: 'rgb(54, 162, 235)' }}
            }};

            var trace3 = {{
                x: fechas,
                y: pulsaciones,
                mode: 'lines+markers',
                name: 'Pulso',
                line: {{ color: 'rgb(255, 206, 86)' }}
            }};

            var data = [trace1, trace2, trace3];

            var layout = {{
                title: 'Evolución de Mediciones',
                xaxis: {{ title: 'Fecha' }},
                yaxis: {{ title: 'Valor' }},
                barmode: 'group'
            }};

            Plotly.newPlot('grafico', data, layout);
        </script>
    </body>
    </html>
    """


def generate_static_plot(fechas, sistolica, diastolica, pulsaciones):
    trace1 = go.Bar(x=fechas, y=sistolica, name='Presión Sistólica', marker=dict(color='rgb(255, 99, 132)'))
    trace2 = go.Bar(x=fechas, y=diastolica, name='Presión Diastólica', marker=dict(color='rgb(54, 162, 235)'))
    trace3 = go.Scatter(x=fechas, y=pulsaciones, mode='lines+markers', name='Pulso',
                        line=dict(color='rgb(255, 206, 86)'))

    data = [trace1, trace2, trace3]

    layout = go.Layout(
        title='Evolución de Mediciones',
        xaxis=dict(title='Fecha'),
        yaxis=dict(title='Valor'),
        barmode='group'
    )

    fig = go.Figure(data=data, layout=layout)

    img_dir = "static/images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_path = os.path.join(img_dir, "grafico.png")
    fig.write_image(img_path)

    return img_path