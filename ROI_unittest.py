#!sqlenv/bin/python

import unittest
import ROI
import json

#import simplejson as json

class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.app = ROI_psycopg2.app.test_client()

    def test_1_list_ROI_dataset(self):
    
        #test get a list of ROI dataset -> OK 200
        response = self.app.get('/datasets/ROI')
        self.assertEqual(json.loads(response.data),{"datasets":["test"]})

    
    def test_2_retrieve_a_dataset(self):
    
        # test get a ROI dataset
        response = self.app.get('/datasets/ROI/test')
        expect = { "ROI":[  {"density": 1,"range": {"east": "3","north": "6","south": "3","west": "0"},"rid": 1,"score": 1.1},
                            {"density": 2,"range": {"east": "6","north": "6","south": "3","west": "3"},"rid": 2,"score": 2.2},
                            {"density": 3,"range": {"east": "3","north": "3","south": "0","west": "0"},"rid": 3,"score": 3.3},
                            {"density": 4,"range": {"east": "6","north": "3","south": "0","west": "3"},"rid": 4,"score": 4.4} ],
                    "name": "test"}
        self.assertEqual(json.loads(response.data),expect)
        
        # test get not exist ROI dataset
        response = self.app.get('/datasets/ROI/error')
        expect = {"ROI": [{"density": -1,"range": {"east": "0","north": "0","south": "0","west": "0"},"rid": -1,"score": -1}],"name": "error"}
        self.assertEqual(json.loads(response.data),expect)
        
        # test get a ROI in ROI dataset
        response = self.app.get('/datasets/ROI/test?rid=1')
        expect ={
                  "ROI": [
                    {
                      "density": 1,
                      "range": {
                        "east": "3",
                        "north": "6",
                        "south": "3",
                        "west": "0"
                      },
                      "rid": 1,
                      "score": 1.1
                    }
                  ],
                  "name": "test"
                }
        self.assertEqual(json.loads(response.data),expect)
        
        # test get a ROI with wrong rid -> error 400
        response = self.app.get('/datasets/ROI/test?rid=-5')
        expect = {"ROI": [{"density": -1,"range": {"east": "0","north": "0","south": "0","west": "0"},"rid": -1,"score": -1}],"name": "error"}
        self.assertEqual(json.loads(response.data),expect)
    
    def test_3_creat_roi_dataset(self):
        # test insert new ROI dataset -> OK 200
        response = self.app.post('/datasets/ROI',data='{"name":"new_table"}', headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,200)
        
        # test insert exist ROI dataset -> error 400
        response = self.app.post('/datasets/ROI',data='{"name":"new_table"}', headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,400)
    
    def test_4_insert_roi(self):
        #test insert ROI -> OK 200
        response = self.app.post('/datasets/ROI/new_table',data = '{"density": 5,"range": {"east": "-8.788","north": "41.2","south": "41.195","west": "-8.782"},"score": 12.0}',headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,200)
        
        #test insert ROI without density and score -> OK 200
        response = self.app.post('/datasets/ROI/new_table',data = '{"range": {"east": "-8.788","north": "41.2","south": "41.195","west": "-8.782"}}',headers= {'Content-Type': 'application/json'})
        self.assertEqual(response.status_code,200)
        
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
        response = self.app.delete('/datasets/ROI/new_table?rid=1')
        self.assertEqual(response.status_code,200)
        
        ''' Delect no exist roi will not return 400 G_G
        #test delete with wrong rid -> error 400
        response = self.app.delete('/datasets/ROI/new_table?rid=-1')
        self.assertEqual(response.status_code,400)
        '''
    
    def test_7_delete_roi_dataset(self):
        #test delete ROI dataset -> OK 200
        response = self.app.delete('/datasets/ROI/new_table')
        self.assertEqual(response.status_code,200)
        
        #test delete ROI not exist -> error 400
        response = self.app.delete('/datasets/ROI/new_table')
        self.assertEqual(response.status_code,400)
       
if __name__ == '__main__':
    unittest.main()
