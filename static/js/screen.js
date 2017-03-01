function animateConfidence(){
  $('.confidence').each(function(){
    var progress = $(this);

    var percent = progress.data('percent');

    var bar = new ProgressBar.Line(this, {
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


$(document).ready(function(){
  window.sr = ScrollReveal();
  sr.reveal('.reveal');
});

$(document).ready(function(){
  $('#fileupload').fileupload({
    url: '/api/v1/classify',
    dataType: 'json',
    progressall: function(e, data) {
      var progress = parseInt(data.loaded / data.total * 100, 10);
      $('#progress .progress-bar').css('width', progress + '%');
    },
    done: function (e, data) {
      $('#result').empty();
      $('#progress').hide();
      $('#progress .progress-bar').css('width', '0%');

      $.each(data.result, function(index, image) {
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
  })
  .on('change', function(e){
    $('#files').empty();
    $('#result').empty();
    $('#progress').show();

    var files = e.target.files;
    var f = files[0];
    var reader = new FileReader();

    reader.onload = (function(file) {
      return function(e) {
        var img = $('<img>');
        img.attr('src', e.target.result);
        img.attr('title', file.name);
        img.addClass('img-responsive');
        img.appendTo('#files');
      };
    })(f);

    reader.readAsDataURL(f);
  })
  .prop('disabled', !$.support.fileInput).parent().addClass($.support.fileInput ? undefined : 'disabled');
});
