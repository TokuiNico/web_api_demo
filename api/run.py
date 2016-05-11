from flask import Flask, jsonify
from flask import abort
from flask import make_response
from flask import request

from dataset import dataset_page
from pats import pats_page
from parameter import parameter_page
from newcact import cact_page

	
app = Flask(__name__)
app.register_blueprint(dataset_page, url_prefix='/datasets')
app.register_blueprint(pats_page, url_prefix='/algo')
app.register_blueprint(parameter_page, url_prefix='/algo')
app.register_blueprint(cact_page)


@app.errorhandler(404)
def not_found():
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
	app.run(port=5566,threaded=True,debug=True)