let mEnter = 1;
let mLeave = 0;
$(document).ready(function () {
    
    $('.showBalloon').hover(function (e) {
        setHover($(this), mLeave);
    }, function (e) {
        setHover($(this), mEnter);
    })
    
});


function setHover(el, add) {
    if (add == 1) {
        let el_index = el.index();
        if (select_index == el_index) {
            el.removeClass('hover');
        }
        el.addClass('balloon');
        
    } else {
        el.removeClass('balloon');
        
        if (select_index == el.index()) {
            el.addClass('hover');
        }
    }
}