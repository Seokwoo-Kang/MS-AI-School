$(document).ready(function () {
    let $lists = $('nav ul');
    let $lists_copy = $lists.clone();
    let $close_el = $('<a href="#">close</a>');

    $lists_copy.css({
        width: 100 + '%',
        height: 100 + '%',
        position: 'absolute',
        top: 0,
        left: 0,
        'text-align': 'center',
        background: 'pink',
        display: 'none'
    });
    $lists_copy.children('li').css({
        'float': 'none',
        padding: '30px',
    });

    $close_el.css({
        position: 'absolute',
        top: '20px',
        right: '20px',
        background: 'white',
        color: 'pink'
    });
    $lists_copy.append($close_el);
    $lists_copy.appendTo('body');

    $close_el.on('click', function (e) {
        $lists_copy.hide();
    });

    $('header div a').on('click', function (e) {
        $lists_copy.show();
    });
});