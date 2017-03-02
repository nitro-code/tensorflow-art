var classifyUrl = '/api/v1/classify';

function resetFileUpload(){
  $('#progress .progress-bar').css('width', '0%');
  $('#progress').hide();
  $('#files').empty();
}

function startFileUpload(){
  $('#files').empty();
  $('#progress').show();
}

function resetClassification(){
  $('#result').empty();
}

function animateConfidence(){
  $('.confidence').each(function(){
    var progress = $(this);
    var percent = progress.data('percent');

    var bar = new ProgressBar.Line(this, {
      color: '#000',
      strokeWidth: 4,
      easing: 'easeInOut',
      duration: 1400,
      trailColor: '#eee',
      trailWidth: 1,
      svgStyle: {width: '100%', height: '100%'},
      text: {
        style: null,
        autoStyleContainer: false
      },
      step: (state, bar) => {
        bar.setText(Math.round(bar.value() * 100) + ' %');
      }
    });

    bar.animate(percent / 100);
  });
}

function printClassificationResult(data){
  $.each(data, function(index, image) {
    $.each(image, function(index, artist) {
      var percent = Math.round(artist.probability * 1000) / 10;
      var description = artist.description.toUpperCase();

      $('<p/>').text(description).appendTo('#result');

      var confidence = $('<div>');
      confidence.addClass('confidence');
      confidence.data('percent', percent);
      confidence.appendTo('#result');
    });
  });

  animateConfidence();
}

function onClassifyClick(id){
  resetFileUpload();
  resetClassification();

  $.ajax({
    url: classifyUrl,
    context: document.body,
    data: {'id': id}
  }).done(function(data) {
    var json = $.parseJSON(data);
    printClassificationResult(json);
  });
}


$(document).ready(function(){
  window.sr = ScrollReveal();
  sr.reveal('.reveal');
});

$(document).ready(function(){
  $('#fileupload').fileupload({
    url: classifyUrl,
    dataType: 'json',
    send: function (e, data) {
      resetClassification();
      startFileUpload();

      var f = data.files[0];
      var reader = new FileReader();

      reader.onload = (function(file) {
        return function(e) {
          var img = $('<img>');
          img.attr('src', e.target.result);
          img.attr('title', file.name);
          img.addClass('img-painting');
          img.addClass('center-block');
          img.appendTo('#files');
        };
      })(f);

      reader.readAsDataURL(f);
    },
    progressall: function(e, data) {
      var progress = parseInt(data.loaded / data.total * 100, 10);
      $('#progress .progress-bar').css('width', progress + '%');
    },
    done: function (e, data) {
      resetFileUpload();
      printClassificationResult(data.result);
    }
  })
  .prop('disabled', !$.support.fileInput).parent().addClass($.support.fileInput ? undefined : 'disabled');
});

$(document).ready(function(){
  $('#btn-painting-1').on('click', function(){
    onClassifyClick(0);
  });

  $('#btn-painting-2').on('click', function(){
    onClassifyClick(1);
  });
});
