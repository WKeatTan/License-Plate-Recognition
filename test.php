<?php 
  if (!empty($_SERVER['HTTP_CLIENT_IP'])) {
    $ip['HTTP_CLIENT_IP'] = $_SERVER['HTTP_CLIENT_IP'];
  }
  //whether ip is from the proxy  
  if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
    $ip['HTTP_X_FORWARDED_FOR'] = $_SERVER['HTTP_X_FORWARDED_FOR'];
  }
  //whether ip is from the remote address  
  if ($_SERVER['REMOTE_ADDR']) {
    $ip['REMOTE_ADDR'] = $_SERVER['REMOTE_ADDR'];
  }

  header('Content-type: application/json');
  echo json_encode($ip);
?>
