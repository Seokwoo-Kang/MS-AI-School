let selected_index = -1;
$(document).ready(function () {
    $('.header-content').on('click', function (e) {
        let index = $(this).index();
        $(this).addClass('select');
        //아무것도 선택되어 있지 않을 때
        if (selected_index == -1) {
            $('.content').eq(index / 2).show();
            selected_index = index;
        } //이전에 선택된 것과 다른 것을 선택했을 때
        else if (selected_index != index) {
            hideElements(selected_index)
            $('.content').eq(index / 2).show();
            selected_index = index;
        } else { //같은 것을 또 선택했을 때
            hideElements(selected_index)
            selected_index = -1;
        }
    })
});

function hideElements(selected_index){
    $('.content').eq(selected_index / 2).hide();
    $('.header-content').eq(selected_index / 2).removeClass('select');
}