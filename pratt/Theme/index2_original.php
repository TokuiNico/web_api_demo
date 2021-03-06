<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!--<meta name="description" content="ADSL Web Demo">-->
    <meta name="author" content="Alvarez.is - BlackTie.co">
    <link rel="shortcut icon" href="assets/ico/favicon.png">

    <title>ADSL Web Demo</title>

    <!-- Bootstrap core CSS -->
    <link href="assets/css/bootstrap.css" rel="stylesheet">

    <!-- Custom styles for this template -->
    <link href="assets/css/main.css" rel="stylesheet">
    
    <link href='http://fonts.googleapis.com/css?family=Lato:300,400,700,300italic,400italic' rel='stylesheet' type='text/css'>
    <link href='http://fonts.googleapis.com/css?family=Raleway:400,300,700' rel='stylesheet' type='text/css'>
    
    <!--<script src="assets/js/jquery.min.js"></script>-->
	<script src="//ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>
    <script src="assets/js/smoothscroll.js"></script>
	<script language="javascript" type="text/javascript" src="jquery.js"></script>
    <script>
    	$(document).ready(function() {
			$("#submit").click(function() {				 
				if(algo!="" & dataset!=""){
				    var optDiv = document.getElementById('options');
					var childDivs = optDiv.getElementsByTagName('div');
					var param = "";
					for( i=0; i< childDivs.length; i++ ){
						//var childDiv = childDivs[i];
						var slider = document.getElementById(parameterlist[i]);
						//alert(slider.value);
						if(i==0)
							param += (parameterlist[i] + "=" + slider.value);
						else{
							param += ( "," + parameterlist[i] + "=" + slider.value);
						}		
					}
					URI=param;
					//alert(param);
					transfer();
				}
				else{
					alert("Please select an algorithm and a dataset!");
				}

			});
		});
    </script>
	
    <script type="text/javascript">
		algo = "";
		dataset ="";
		var parameterlist = [];
		URI = "";
		
	    function algoOnChange(x){
			//var html = []
			//$("#result").html(html.join('')).css("background-color", "#B3D1FF");
			var index = 0;
			var alldataset = document.getElementById("datalist");
			algo = x;

			if(x=="PATS"){
          		<?php
              //$str =  file_get_contents('http://127.0.0.1:5566/datasets/ROI');
            		$url = 'http://127.0.0.1:5566/algo/show/PATS';
                	$str =  file_get_contents($url);
          		?>
          		var data = <?php echo json_encode($str)?>;
        	}
        	else if(x=="CACT"){

          		<?php
              //$str =  file_get_contents('http://127.0.0.1:5566/datasets/ROI');
             		$url = 'http://140.113.86.130/~anna/CACT_parameter.json';
                	$str =  file_get_contents($url);
          		?>
          		var data = <?php echo json_encode($str)?>;

        	}
            
            
            var obj = jQuery.parseJSON(data);
            
            document.getElementById('options').innerHTML = "";//clean div
            parameterlist = [];

            for (var key in obj) {
                
                if(key=="Parameter"){
					for (var each in obj[key]){

						var newDiv = document.createElement("div");
						newDiv.setAttribute("class","row");
						
						
						var newLabel = document.createElement("label");
						newLabel.setAttribute("style","margin:1px;font-size:12pt;color:Black;");
						newLabel.setAttribute("class","col-lg-2");
						newLabel.innerHTML = obj.Parameter[each].name + ":";
						newDiv.appendChild(newLabel)
						
						var slider = document.createElement('input');
						slider.id = obj.Parameter[each].name;
						slider.type = 'range';
						slider.setAttribute("class","col-lg-7");
						slider.setAttribute("min", obj.Parameter[each].min);
						slider.max = obj.Parameter[each].max;

						if(obj.Parameter[each].type=="int")	{						  
							slider.step = 1;
							slider.value = obj.Parameter[each].default;
						}
						else{						  
						    slider.step = 0.00001;
						    slider.value = obj.Parameter[each].default;
						}
						slider.setAttribute("oninput","sliderOnchange(this.id,this.value);");
						//slider.setAttribute("style","margin-top:4px;");
						newDiv.appendChild(slider)
						
						var valueLabel = document.createElement("label");
						valueLabel.setAttribute("style","margin:1px;font-size:10pt;color:grey;");
						valueLabel.setAttribute("class","col-lg-3");
						valueLabel.id = obj.Parameter[each].name + "_value_id";
						valueLabel.innerHTML = obj.Parameter[each].default;
				 
						newDiv.appendChild(valueLabel)
						document.getElementById('options').appendChild(newDiv);
						parameterlist[parameterlist.length] = obj.Parameter[each].name;

					}
                }
                else if(key=="Dataset"){
                    var i = 0;
                    $('#datalist').find('option').remove();

                    for(var name in obj.Dataset)
                    {
						AddOpt = new Option(obj.Dataset[name]);
						alldataset.options[i] = AddOpt;
						i++;
                    }  
                }

				       
			}            
        }

        function dataOnChange(x){
			
			dataset = x;
        }

		function sliderOnchange(id,value){
			
			vid = id+ "_value_id";
			document.getElementById(vid).innerHTML = value;
		}

        function transfer(){
			//alert("sending...");
			var html = [];
			html.push("Please wait.....");
			$("#result").html(html.join('')).css("background-color", "#B3D1FF");
        

	        var link="getJSON.php?value=" + URI + "&algo=" + algo + "&dataset=" + dataset; 
	        
	        $.ajaxSetup({
	        	timeout:1200000 // in milliseconds 
	        });

	        //alert(link);

	        $.get(link,function(data) {

            	var html = [];
                alert(data);
            	var obj = JSON.parse(data);
            	var Colors = [];
        		var ramColor;

        		if(algo=="PATS"){
          			for (var i = 0; i < obj.candidate_tid.length; i++) {

	            		do {
	                		ramColor = '#' + Math.random().toString(16).slice(2, 8).toUpperCase();
	            		}while (ramColor in Colors);

	            		var tid = obj.candidate_tid[i].tid;
	            		var score = obj.candidate_tid[i].score;
	            		var size = obj.candidate_tid[i].points.length;
	            		var timestart =  new Date(obj.candidate_tid[i].points[0].timestamp);
			            var timefinish = new Date(obj.candidate_tid[i].points[size-1].timestamp);
			            var offset = (timefinish.getTime()/1000) - (timestart.getTime()/1000);
			            var dt = new Date(offset);
			            var str = "duration : " +  Math.floor(offset/(60*60)) + ":" + Math.floor(offset/(60)) +":" + offset%60 + '<br>';
			            
	            		Colors.push(ramColor);
			            obj.candidate_tid[i].color = ramColor;
			            html.push("tid:" + tid,"<br>");
			            html.push("score:" + score,"<br>");
			            html.push("start:" + obj.candidate_tid[i].points[0].lat + "," + obj.candidate_tid[i].points[0].lon + "," + obj.candidate_tid[i].points[0].timestamp,"<br>");
			            html.push("end:" + obj.candidate_tid[i].points[size-1].lat + "," + obj.candidate_tid[i].points[size-1].lon + "," + obj.candidate_tid[i].points[size-1].timestamp,"<br>");
			            html.push("color:" + "<font color=" + obj.candidate_tid[i].color + ">"+obj.candidate_tid[i].color+"</font>","<br>");
			            html.push(str,"<br>");
	          		}
	        	}
	        	else{

		          	for(var i= 0; i<obj.trajectories.length; i++)
		          	{
		            	do {
		                	ramColor = '#' + Math.random().toString(16).slice(2, 8).toUpperCase();
		            	}
			            while (ramColor in Colors);
			            Colors.push(ramColor);
			            obj.trajectories[i].color = ramColor;
			            var size = obj.trajectories[i].points.length;
			            var tid = obj.trajectories[i].tid;
			            html.push("tid:" + tid,"<br>");
			            html.push("start:" + obj.trajectories[i].points[0].lat + "," + obj.trajectories[i].points[0].lon,"<br>");
			            html.push("end:" + obj.trajectories[i].points[size-1].lat + "," + obj.trajectories[i].points[size-1].lon,"<br>");
			            html.push("total points:" + size,"<br>");
			            html.push("color:" + "<font color=" + obj.trajectories[i].color + ">"+obj.trajectories[i].color+"</font>","<br>");
			            html.push("","<br>");
		          	}

		          	var centerpoint = new google.maps.LatLng(24.795917, 120.997874);
		          	map.setCenter(centerpoint);
		        }
				
				$("#result").html(html.join('')).css("background-color", "#B3D1FF");
	           	draw(obj);

	        });
    	}
		
        
    	var flightPathList = [];
    	var rectangle;

		function draw(obj){
			if(algo=="PATS"){
				var bounds = {
		            north: obj.range.north,
		            south: obj.range.south,
		            east: obj.range.east,
		            west: obj.range.west
		        };
	        
		        if (typeof rectangle != 'undefined')
		            rectangle.setMap(null);
	   
		        rectangle = new google.maps.Rectangle({
		            bounds: bounds,
		            editable: true,
		            draggable: true
		        });

	        	rectangle.setMap(map);
        
		        //Plot trajectory
		        
		        if (flightPathList.length != 0){
		            for (var i = 0; i < flightPathList.length; i++)
		            {
		                flightPathList[i].setMap(null);
		            }
		            
		            flightPathList = []
		        }


		        for (var tra_idx = 0; tra_idx < obj.candidate_tid.length; tra_idx=tra_idx+1){
		            //Get points of trajectory
		            var points = obj.candidate_tid[tra_idx].points;
		            var len = points.length;
		            var flightPlanCoordinates = [];
		            var tid = obj.candidate_tid[tra_idx].tid;
		 
		            //Set Polyline
		            for (var i = 0; i < len-1; i=i+1){
		                flightPlanCoordinates.push({lat: points[i].lat, lng: points[i].lon});
		            }
		            
		            //Create a new Polyline object
		            var flightPath = new google.maps.Polyline({
		                path: flightPlanCoordinates,
		                geodesic: true,
		                strokeColor: obj.candidate_tid[tra_idx].color,
		                strokeOpacity: 1.0,
		                strokeWeight: 2
		                });
		            
		            	google.maps.event.addListener(flightPath,'click', function() {

		                for (var j = 0; j < obj.candidate_tid.length; j++) {
		                  if(obj.candidate_tid[j].tid==tid){
		                     $('#marker-tooltip').html("score:" + obj.candidate_tid[j].score + '<br>').css({
		                          'left': cursorX,
		                          'top': cursorY
		                      }).show();
		                  }	                  
		                }
		            }); 

		            //PLOT!
		            flightPath.setMap(map);
		                // Add an event listener on the rectangle.
		            rectangle.addListener('bounds_changed', showNewRect);
		            // Define an info window on the map.
		            infoWindow = new google.maps.InfoWindow();
		        }
		    }
		    else{//CACT
		        if (flightPathList.length != 0){
		            for (var i = 0; i < flightPathList.length; i++){
		                flightPathList[i].setMap(null);
		            }         
		            flightPathList = []
		        }
		        
		        for (var tra_idx = 0; tra_idx < obj.trajectories.length; tra_idx=tra_idx+1){
		            //Get points of trajectory
		            var points = obj.trajectories[tra_idx].points;
		            var len = points.length;
		            var flightPlanCoordinates = [];
		            var tid = obj.trajectories[tra_idx].tid;
		 
		            //Set Polyline
		            for (var i = 0; i < len-1; i=i+1){
		                flightPlanCoordinates.push({lat: points[i].lat, lng: points[i].lon});
		            }
		            
		            //Create a new Polyline object
		            var flightPath = new google.maps.Polyline({
		                path: flightPlanCoordinates,
		                geodesic: true,
		                strokeColor: obj.trajectories[tra_idx].color,
		                strokeOpacity: 1.0,
		                strokeWeight: 2
		                });
		            flightPath.setMap(map);
		        }
		    }
        }

        function showNewRect(event) {
	      	var ne = rectangle.getBounds().getNorthEast();
	      	var sw = rectangle.getBounds().getSouthWest();
	       
	      	east = rectangle.getBounds().getNorthEast().lng();
	      	north = rectangle.getBounds().getNorthEast().lat();
	      	west = rectangle.getBounds().getSouthWest().lng();
	      	south = rectangle.getBounds().getSouthWest().lat();
	      
	     	document.getElementById("east_value_id").innerHTML = east;
	    	document.getElementById("north_value_id").innerHTML = north;
	    	document.getElementById("west_value_id").innerHTML = west;
	    	document.getElementById("south_value_id").innerHTML = south;
	      
	    	document.getElementById("east").value = east;
	    	document.getElementById("west").value = west;
	    	document.getElementById("north").value = north;
	    	document.getElementById("south").value = south;
	      
	      // var contentString = '<b>Rectangle moved.</b><br>' +
	          // 'New north-east corner: ' + ne.lat() + ', ' + ne.lng() + '<br>' +
	          // 'New south-west corner: ' + sw.lat() + ', ' + sw.lng();

	      // Set the info window's content and position.
	      // infoWindow.setContent(contentString);
	      // infoWindow.setPosition(ne);

	      // infoWindow.open(map);
	    }
    </script>
	
	<style>
		#headerwrap{
			height: 950px;
		}
		#welcome{
			margin-top: 280px;
		}
		#map {
			height: 600px;
			margin-bottom: 15px;
			padding-left: 15px;
		}
		input[type="range"]{
			width: 200px;
		}
		.box-title {
			color:#000;
			background-color:#B3D1FF;
			font-size: 16px;
			padding: 6px 0 10px 10px;
		}
		#result{
			color:#000;
			height: 600px;
			overflow: auto;
			background-color:#B3D1FF;
		}
		#c2 {
			padding-left: 15px;
			//height: 850px;
			//overflow: auto;
		}
		#PATS{
			padding-left: 5px;
		}
		#CACT{
			padding-left: 5px;
		}
		#submit{
			background-color:#E69409;
			//background-color:#3878C7;
		}
		<!--
		#b1{
			height: 160px;
		}
		#b2{
			height: 160px;
		}
		#b3{
			height: 160px;
		}
		-->
	</style>
  </head>

  <body data-spy="scroll" data-offset="0" data-target="#navigation">

    <!-- Fixed navbar -->
	    <div id="navigation" class="navbar navbar-default navbar-fixed-top">
	      <div class="container">
	        <div class="navbar-header">
	          <button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".navbar-collapse">
	            <span class="icon-bar"></span>
	            <span class="icon-bar"></span>
	            <span class="icon-bar"></span>
				<span class="icon-bar"></span>
	          </button>
	          <a class="navbar-brand" href="#"><b>ADSL</b></a>
	        </div>
	        <div class="navbar-collapse collapse">
	          <ul class="nav navbar-nav">
	            <li class="active"><a href="#home" class="smothscroll">Home</a></li>
	            <li><a href="#map_block" class="smothscroll">Map</a></li>
	            <li><a href="#report_block" class="smothScroll">Report</a></li>
	            <li><a href="#contact" class="smothScroll">Contact</a></li>
	          </ul>
	        </div><!--/.nav-collapse -->
	      </div>
	    </div>


  <section id="home" name="home"></section>
	<div id="headerwrap">
	    <div class="container" id="welcome">
	    	<div class="row centered">
	    		<div class="col-lg-12">
					<h1>Welcome To <b>ADSL</b></h1>
					<h3>The coolest trajectories mining demo system</h3>
					<br>
	    		</div>
	    		
	    	
	    	</div>
	    </div> <!--/ .container -->
	</div><!--/ #headerwrap -->


	
	
	
  <section id="map_block" name="map_block"></section>
	<script>
        var map;
        function initMap() {
        map = new google.maps.Map(document.getElementById('map'), {
        center: {lat: 41.215728, lng: -8.627697},
        zoom: 12
        });
      }
    </script>
	
    <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyA9u0gJyOELtFqHuOMPCo_QSl0lvF3Rlv0&callback=initMap"
    async defer></script>

	<div class="container" id="map_title">
		<div class="row centered">
			<br>
			<br>
			<h1>Demo</h1>
			<br>

		</div>
	</div>
	<div class="container" id="map_map">
		<div class="row">		  
		  <div class="col-lg-9" id="map"></div>
		  <div class="col-lg-3">
		    <div id="result"></div>
		  </div>
		</div>
		
		<div class="row">
		  <div class="col-lg-2" id="b1">
		    <div class="box-title">Algorithm</div>
			<select id="algoList"  size="5" style="font-size:13pt;height:80%;width:100%;" onchange="algoOnChange(this.options[this.options.selectedIndex].value)">		                
			  <option id="PATS">PATS</option>
			  <option id="CACT">CACT</option>			  
			</select>	
		  </div>
		  
		  <div class="col-lg-2" id="b2">
		    <div class="box-title">Dataset</div>
			<select id="datalist"  size="5" style= "font-size:13pt;height:80%;width:100%;" onchange="dataOnChange(this.options[this.options.selectedIndex].value)"> 
			</select>
		  </div>
		  
		  <div class="col-lg-4" id="b3">
		    <div class="box-title">Options</div>
			<div id="options"></div>
		  </div>
		  
		  <div class="col-lg-1">
		    <button id="submit" type="submit" class="btn btn-outline-inverse">Submit</button>
			
		  </div>		  	
		</div>
		<br>
		<br>
	</div> <!--/ .container -->



	
	
	
  <section id="report_block" name="report_block"></section>
    <div id="report_block">
		<div class="container" id="report_title">
			<div class="row centered">				
				<hr>
				<br>
				<h1>Visual Report of Trajectories</h1>
				
			</div>
		</div>
		<div class="container" id="report_report">

			<div class="row">

				<div class="col-lg-12" id="c2"></div>
			</div>
			<br>
			<br>
			<br>

		</div>	
	</div>

  <section id="contact" name="contact"></section>	
	<div id="footerwrap">
		<div class="container">
			<div class="col-lg-5">
				<h3>Address</h3>
				<p>
				Engineering Building 3, Room 621,<br/>
				University Road 1001, NCTU,<br/>
				Hsinchu City,<br/>
				30010,<br/>
				Taiwan
				</p>
				
			</div>
			<div class="col-lg-7">
				<h3>Contacts</h3>
				<p>
				ChyuLin: vegetable80923@gmail.com<br/>
				David: haodongdavidwu@gmail.com<br/>
				Anna: dearannaiam@gmail.com
				</p>
			</div>
		</div>
	</div>
	
	<script src="https://public.tableau.com/javascripts/api/tableau-2.js"></script>

	<script>
		var vizDiv = document.getElementById('c2');
		var vizURL = 'https://public.tableau.com/views/tra3/TaxiTraVis';
		var options = {
			//height: '450px',
			//width: '700px',
			hideTabs: true,
			hideToolbar: true
		}
		viz = new tableauSoftware.Viz(vizDiv, vizURL, options);
	</script>

	


    <!-- Bootstrap core JavaScript
    ================================================== -->
    <!-- Placed at the end of the document so the pages load faster -->
    <script src="assets/js/bootstrap.js"></script>
	<script>
	$('.carousel').carousel({
	  interval: 3500
	})
	</script>
  </body>
</html>
