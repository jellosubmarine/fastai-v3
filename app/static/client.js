var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

// function showPicked(input) {
//   el("upload-label").innerHTML = input.files[0].name;
//   var reader = new FileReader();
//   reader.onload = function(e) {
//     el("image-picked").src = e.target.result;
//     el("image-picked").className = "";
//   };
//   reader.readAsDataURL(input.files[0]);
// }

function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
           true);

  xhr.onreadystatechange = function() { console.log(xhr.status); }

                           xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    var response = e.target.response;
    var jsonResponse = JSON.parse(response);
    console.log(jsonResponse);
    console.log('Response on ' + jsonResponse['image']);
    if (jsonResponse['image'] !== undefined) {
      var encodedImage = jsonResponse['image'];
      el("image-picked").src = "data:image/png;base64," + encodedImage;
      el("image-picked").className = "";
      el("result-label").innerHTML = `Result = ${jsonResponse["result"]}`;
      //  var response = JSON.parse(e.target.responseText);
      //
      // el("debug-label").innerHTML = `Debug = ${response["debug"]}`;
    }
    el("analyze-button").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}
