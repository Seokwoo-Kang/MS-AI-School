$(document).ready(function () {
    let num_array=[];
    let op_array=[];
    let temp_array=[];

    let screen = $('.screen');

    $('button.num').click(function (e) {
        let num_s = $(this).val();
        let num_i = parseInt(num_s);
        num_array.push(num_i);
        screen_dis(num_s);
    });

    $('button.op').click(function(){
        let len_num = num_array.length;
        let origin_num = 0;

        if(len_num > 0 ) {
            for (let i = 0; i < len_num; i++) {
                let index = len_num - 1 - i;
                origin_num += Math.pow(10, index) * num_array[i];
            }

            temp_array.push(origin_num);
            num_array = [];
        }

        let op_s = $(this).val();
        op_array.push(op_s);

        if(op_s == '='){
            let result = cal(temp_array[0], temp_array[1], op_array[0]);;
            op_array.pop();
            for(let i = 1; i < op_array.length; i ++){
                result = cal(result, temp_array[i+1], op_array[i]);
            }
            screen_dis('all');
            screen_dis(result.toString());

            temp_array=[];
            temp_array.push(result);

            op_array=[];


        }else if(op_s == 'clear'){
            screen_dis('all');
            num_array=[];
            temp_array=[];
            op_array=[];
        }else {
            screen_dis(op_s);
        }
    });

    function cal(num1, num2, op){
        let result=0;
        if(op == '+'){
            result = num1 + num2;
        }else if(op=='-'){
            result = num1 - num2;
        }else if( op == '*'){
            result = num1 * num2;
        }else {
            result = num1 / num2;
        }
        return result;
    }

    function screen_dis(el){
         let screen_text_value = screen.text();
         screen.text(screen_text_value + el);

         if(el == 'all'){
             screen.text('');
         }
    }
})