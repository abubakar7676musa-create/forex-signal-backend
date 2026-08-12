-- Reference schema. The FastAPI app creates these automatically via SQLAlchemy on startup
-- (Base.metadata.create_all). This file is provided for manual review, DBA setup, or
-- environments where you prefer to provision the schema before first app boot.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE user_role AS ENUM ('user', 'admin');

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    firebase_uid VARCHAR(128) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    role user_role NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    fcm_token VARCHAR(512),
    favorite_pairs TEXT[] DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_firebase_uid ON users(firebase_uid);

CREATE TYPE signal_direction AS ENUM ('BUY', 'SELL');
CREATE TYPE signal_status AS ENUM ('ACTIVE', 'HIT_TP1', 'HIT_TP2', 'HIT_SL', 'EXPIRED', 'CANCELLED');

CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pair VARCHAR(20) NOT NULL,
    direction signal_direction NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    stop_loss DOUBLE PRECISION NOT NULL,
    take_profit_1 DOUBLE PRECISION NOT NULL,
    take_profit_2 DOUBLE PRECISION NOT NULL,
    risk_reward_ratio DOUBLE PRECISION NOT NULL,
    confidence_score INTEGER NOT NULL,
    timeframe VARCHAR(10) DEFAULT '1h',
    status signal_status DEFAULT 'ACTIVE',
    explanation TEXT,
    confirmations TEXT,
    is_published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_signals_pair ON signals(pair);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
CREATE INDEX IF NOT EXISTS idx_signals_published ON signals(is_published);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at DESC);

CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    body TEXT NOT NULL,
    signal_id UUID,
    is_broadcast BOOLEAN DEFAULT TRUE,
    target_user_id UUID,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS daily_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date TIMESTAMP UNIQUE NOT NULL,
    total_signals INTEGER DEFAULT 0,
    buy_signals INTEGER DEFAULT 0,
    sell_signals INTEGER DEFAULT 0,
    avg_confidence DOUBLE PRECISION DEFAULT 0,
    signals_hit_tp1 INTEGER DEFAULT 0,
    signals_hit_tp2 INTEGER DEFAULT 0,
    signals_hit_sl INTEGER DEFAULT 0,
    new_users INTEGER DEFAULT 0,
    active_users INTEGER DEFAULT 0,
    most_active_pair VARCHAR(20)
);
