<?php   
	ini_set('default_socket_timeout', 900);//900s
    $value =  $_GET['value'];
    $algo =  $_GET['algo'];
    $dataset =  $_GET['dataset'];
    $param = str_replace(",","&",$value);
    if ($algo=="PATS") {
	  $url = 'http://127.0.0.1:5566/algo/'.$algo.'/'.$dataset.'?'.$param;
	}
	else{
		$url = 'http://127.0.0.1:5005/api/cact/cact'.'?'.$param;
	}
    $str =  file_get_contents($url);
    //echo $url.'<br>';
    echo $str;
?> 