let select_index = 0;
$(document).ready(function () {
    setSelect(select_index);
    $('ul li').hover(function (e) {
        setHover($(this), 1);
    }, function (e) {
        setHover($(this), 0);
    })
    $('ul li').click(function (e) {
        setSelect($(this).index());
    })
});

function setSelect(index) {
    $('ul li').eq(index).addClass('select');
    $('.content').eq(index).addClass('select').show();

    if (index != select_index) {
        $('ul li').eq(select_index).removeClass('select');
        $('.content').eq(select_index).hide();
        select_index = index;
    }
}

function setHover(el, add) {
    if (add == 1) {
        var el_index = el.index();
        if (select_index == el_index) {
            el.removeClass('select');
        }
        el.addClass('hover');
        el.animate({
            height: '75px',
            'line-height': '60px'
        });
    } else {
        el.removeClass('hover');
        el.animate({
            height: '40px',
            'line-height': '35px'
        });
        if (select_index == el.index()) {
            el.addClass('select');
        }
    }
}