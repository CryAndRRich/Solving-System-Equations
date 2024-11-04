import base64
from flask import Flask, render_template, request
from SolveEquations import Solution
from Plot import Plot
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/plot', methods=['POST'])
def plot():
    ind = 1
    matrix = []
    while True:
        try:
            entry = request.form['entry' + str(ind)]
            matrix.append(entry)
            ind += 1
        except:
            break
    
    if len(Solution(matrix)) == 1:
        return render_template('error.html', msg = Solution(matrix)[0])

    solution_str, solution, A = Solution(matrix)

    img, max_accuracy = Plot(request.form['method'], A, solution)
    img_data = img.getvalue()
    img_base64 = base64.b64encode(img_data).decode('utf-8')

    msg = {'image_data': img_base64,
           'max_accuracy': max_accuracy,
           'solution_str': solution_str}

    return render_template('plot.html', msg = msg)

