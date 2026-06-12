DROP TABLE IF EXISTS device_types;

CREATE TABLE device_types (
    id        INT PRIMARY KEY,        -- Id: идентификатор типа (приходит в событии из Kafka)
    type_name VARCHAR(100) NOT NULL   -- TypeName: наименование типа
);
