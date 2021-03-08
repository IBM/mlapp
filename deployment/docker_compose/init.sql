
CREATE TABLE public.analysis_results (
	model_id uuid NOT NULL,
	asset_name varchar(255) NOT NULL,
	asset_label varchar(255) NULL,
	pipeline varchar(255) NOT NULL,
	properties json NOT NULL,
	metadata json NOT NULL,
	environment varchar(255) NULL,
	created_at timestamptz NULL,
	CONSTRAINT analysis_results_pkey PRIMARY KEY (model_id)
);

CREATE TABLE public.asset_accuracy_monitoring (
	model_id uuid NOT NULL,
	asset_name varchar(255) NOT NULL,
	asset_label_name varchar(255) NOT NULL,
	created_at timestamptz NULL,
	updated_at timestamptz NULL,
	"timestamp" timestamptz NULL DEFAULT CURRENT_TIMESTAMP,
	model_accuracy json NOT NULL
);

CREATE TABLE public.flows (
	flow_id uuid NOT NULL,
	metadata json NOT NULL,
	properties json NOT NULL,
	created_at timestamptz NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.models_history (
	model_id uuid NOT NULL,
	asset_name varchar(255) NOT NULL,
	asset_label varchar(255) NULL,
	created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT models_history_pkey PRIMARY KEY (model_id, created_at)
);

CREATE TABLE public.target (
	"timestamp" timestamptz NOT NULL,
	model_id uuid NOT NULL,
	forecast_id uuid NOT NULL,
	"index" varchar(255) NOT NULL,
	y_true float4 NULL,
	y_hat float4 NULL,
	"type" int4 NOT NULL,
	CONSTRAINT target_pkey PRIMARY KEY (model_id, forecast_id, index, type)
);
