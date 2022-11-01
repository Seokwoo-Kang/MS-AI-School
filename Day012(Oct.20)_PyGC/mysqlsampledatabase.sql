-- 1.
-- SELECT * #선택칼럼명 ex)lastName, firstName, jobTitle
-- -- FROM employees;				#가져온곳

-- #2.역순정렬
-- SELECT
-- 	contactLastname,
-- 	contactFirstname
-- FROM
-- 	customers
-- ORDER BY
-- 	contactLastname DESC;

-- #3
-- SELECT 
--     contactLastname, 
--     contactFirstname
-- FROM
--     customers
-- ORDER BY 
-- 	contactLastname DESC , 
-- 	contactFirstname ASC;

# 4

-- SELECT 
--     orderNumber, 
--     orderlinenumber, 
--     quantityOrdered * priceEach
-- FROM
--     orderdetails
-- ORDER BY 
--    quantityOrdered * priceEach DESC;

# 5

-- SELECT 
--     orderNumber,
--     orderLineNumber,
--     quantityOrdered * priceEach AS subtotal
-- FROM
--     orderdetails
-- ORDER BY subtotal DESC;

# 6

#SELECT FIELD('A', 'A', 'B','C');
#SELECT FIELD('B', 'A','B','C');

# 7

-- SELECT 
--     orderNumber, status
-- FROM
--     orders
-- ORDER BY FIELD(status,
--         'In Process',
--         'On Hold',
--         'Cancelled',
--         'Resolved',
--         'Disputed',
--         'Shipped');

# 8

-- SELECT 
--     firstName, lastName, reportsTo
-- FROM
--     employees
-- ORDER BY reportsTo; #ASC, NULLs

# 9 WHERE 조건 (BETWEEN의 경우 문자도 가능, LIKE문은 과부하를 유발하므로 꼭 필요할때만, IS NULL문 NULL값을 찾기위함, <>연산 아닌 모든것)



# 10 DISTINCT 구별 (중복제거)

# 11 LIMIT 데이터 출력 갯수 제약

# 12 나머지 공부

# 13 JOIN 데이터합(여러조건으로 가능)

-- CREATE DATABASE IF NOT EXISTS salesdb;
-- USE salesdb;
-- CREATE TABLE products (
--     id INT PRIMARY KEY AUTO_INCREMENT,
--     product_name VARCHAR(100),
--     price DECIMAL(13,2 )
-- );

-- CREATE TABLE stores (
--     id INT PRIMARY KEY AUTO_INCREMENT,
--     store_name VARCHAR(100)
-- );

-- CREATE TABLE sales (
--     product_id INT,
--     store_id INT,
--     quantity DECIMAL(13 , 2 ) NOT NULL,
--     sales_date DATE NOT NULL,
--     PRIMARY KEY (product_id , store_id),
--     FOREIGN KEY (product_id)
--         REFERENCES products (id)
--         ON DELETE CASCADE ON UPDATE CASCADE,
--     FOREIGN KEY (store_id)
--         REFERENCES stores (id)
--         ON DELETE CASCADE ON UPDATE CASCADE
-- );
-- INSERT INTO products(product_name, price)
-- VALUES('iPhone', 699),
--       ('iPad',599),
--       ('Macbook Pro',1299);

-- INSERT INTO stores(store_name)
-- VALUES('North'),
--       ('South');

-- INSERT INTO sales(store_id,product_id,quantity,sales_date)
-- VALUES(1,1,20,'2017-01-02'),
--       (1,2,15,'2017-01-05'),
--       (1,3,25,'2017-01-05'),
--       (2,1,30,'2017-01-02'),
--       (2,2,35,'2017-01-05');
SELECT 
    store_name,
    product_name,
    SUM(quantity * price) AS revenue
FROM
    sales										#sales 조건1
        INNER JOIN
    products ON products.id = sales.product_id	#products.id = sales.product_id조건2
        INNER JOIN
    stores ON stores.id = sales.store_id	#stores.id = sales.store_id 조건3
GROUP BY store_name , product_name; 

##

-- SELECT 
--     store_name, product_name
-- FROM
--     stores AS a
--         CROSS JOIN
--     products AS b;

##

-- SELECT 
--     b.store_name,
--     a.product_name,
--     IFNULL(c.revenue, 0) AS revenue
-- FROM
--     products AS a
--         CROSS JOIN
--     stores AS b
--         LEFT JOIN
--     (SELECT 
--         stores.id AS store_id,
--         products.id AS product_id,
--         store_name,
--             product_name,
--             ROUND(SUM(quantity * price), 0) AS revenue
--     FROM
--         sales
--     INNER JOIN products ON products.id = sales.product_id
--     INNER JOIN stores ON stores.id = sales.store_id
--     GROUP BY stores.id, products.id, store_name , product_name) AS c ON c.store_id = b.id
--         AND c.product_id= a.id
-- ORDER BY b.store_name;

-- SELECT*
-- FROM orders

-- SELECT 
--     orderNumber, status
-- FROM
--     orders
-- ORDER BY FIELD(status,
--         'In Process',
--         'On Hold',
--         'Cancelled',
--         'Resolved',
--         'Disputed',
--         'Shipped');

-- SELECT FIELD('A', 'A', 'B','C');
-- SELECT FIELD('B', 'A','C','C','C','B','C');

