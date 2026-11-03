-- Runs once, on first initialisation of the data volume.
-- A read-only role for the postgres MCP tool server: the agent can query its own metadata
-- but cannot write, even if the model emits a mutating statement.
CREATE ROLE pka_ro LOGIN PASSWORD 'pka_ro' NOSUPERUSER NOCREATEDB NOCREATEROLE;
GRANT CONNECT ON DATABASE pka TO pka_ro;
GRANT USAGE ON SCHEMA public TO pka_ro;
-- Tables are created later by alembic (as role pka); grant SELECT on them automatically.
ALTER DEFAULT PRIVILEGES FOR ROLE pka IN SCHEMA public GRANT SELECT ON TABLES TO pka_ro;
ALTER ROLE pka_ro SET default_transaction_read_only = on;
ALTER ROLE pka_ro SET statement_timeout = '5s';

-- Separate database for the integration test suite.
CREATE DATABASE pka_test OWNER pka;
