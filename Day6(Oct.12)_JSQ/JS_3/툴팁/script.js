$(document).ready(function () {
    let balloon = $('.balloon');
    function updateBalloonPosition(x, y) {
        balloon.css({
            left: x + 10,
            top: y
        });
    }
    $('.showBalloon').each(function () {
        let $el = $(this);
        let text = $el.attr('title');

        $el.hover(function (event) {
            balloon.text(text);
            updateBalloonPosition(event.pageX, event.pageY);
            balloon.show();
        }, function (event) {
            balloon.hide();
        });

        $el.mousemove(function (event) {
            updateBalloonPosition(event.pageX, event.pageY);
        })
    });

});