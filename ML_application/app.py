from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch data from the database and prepare for rendering.
    data = {'title': 'My Flask App'}

    # Render the HTML template.
    return render_template('ML_application\check.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)