#!sqlenv/bin/python

import os
import unittest
from flask import Flask, jsonify
import ROI
import simplejson as json

class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = hdwu.app.test_client()

    def test_1_list_ROI_dataset(self):
        response = self.app.get('/datasets/ROI')

        self.assertEqual(response.status_code,200)
    
    def test_2_retrieve_a_dataset(self):
        response = self.app.get('/datasets/ROI/test')
        self.assertEqual(response.status_code,200)
        response = self.app.get('/datasets/ROI/error')
        self.assertEqual(response.status_code,500)
        response = self.app.get('/datasets/ROI/test?rid=2')
        self.assertEqual(response.status_code,200)
        response = self.app.get('/datasets/ROI/test?rid=-5')
        self.assertEqual(response.status_code,400)
        
    def test_3_creat_roi_dataset(self):
        response = self.app.post('/datasets/ROI',data='{"name":"new_table"}', headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,201)
        response = self.app.post('/datasets/ROI',data='{"name":"new_table"}', headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,400)
    
    def test_4_insert_roi(self):
        #test insert ROI -> OK 201
        response = self.app.post('/datasets/ROI/new_table',data = '{"density": 5,"range": {"east": "-8.788","north": "41.2","south": "41.195","west": "-8.782"},"score": 12.0}',headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,201)
        
        #test insert ROI without density and score -> OK 201
        response = self.app.post('/datasets/ROI/new_table',data = '{"range": {"east": "-8.788","north": "41.2","south": "41.195","west": "-8.782"}}',headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,201)
        
        #test insert ROI without range -> error 400
        response = self.app.post('/datasets/ROI/new_table',data = '{"density": 5,"score": 12.0}',headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,400)
    
    def test_5_modify_roi(self):
        #test modify ROI -> ok 201
        response = self.app.put('/datasets/ROI/new_table',data = '{"rid":0, "density": 5,"score": 12.0}',headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,200)
        
        #test modify ROI without rid -> error 400
        response = self.app.put('/datasets/ROI/new_table',data = '{"density": 5,"score": 12.0}',headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,400)
        
        #TODO: test modify ROI with wrong rid -> error 400
        
    def test_6_delete_roi(self):
        #test delete -> OK 200
        response = self.app.delete('/datasets/ROI/new_table?rid=0')
        self.assertEqual(response.status_code,200)
        
        #test delete with wrong rid -> error 400
        response = self.app.delete('/datasets/ROI/new_table?rid=-1')
        self.assertEqual(response.status_code,400)
    
    def test_7_delete_roi_dataset(self):
        #test delete ROI dataset -> OK 200
        response = self.app.delete('/datasets/ROI/new_table')
        self.assertEqual(response.status_code,200)
        
        #test delete ROI not exist -> error 400
        response = self.app.delete('/datasets/ROI/new_table')
        self.assertEqual(response.status_code,400)
        
if __name__ == '__main__':
    unittest.main()
