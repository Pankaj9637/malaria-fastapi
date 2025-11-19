-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.admin (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  auth_user_id uuid NOT NULL UNIQUE,
  name character varying NOT NULL,
  org_id bigint NOT NULL,
  email character varying,
  phone character varying,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT admin_pkey PRIMARY KEY (id),
  CONSTRAINT admin_org_id_fkey FOREIGN KEY (org_id) REFERENCES public.org(id),
  CONSTRAINT admin_auth_user_id_fkey FOREIGN KEY (auth_user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.blood_samples (
  id integer GENERATED ALWAYS AS IDENTITY NOT NULL,
  patient_id integer NOT NULL,
  sample_date date DEFAULT CURRENT_DATE,
  image_path character varying,
  image_metadata character varying,
  processing_status character varying DEFAULT 'pending'::character varying,
  error_message text,
  storage_url text,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT blood_samples_pkey PRIMARY KEY (id),
  CONSTRAINT blood_samples_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES public.patients(id)
);
CREATE TABLE public.doctor (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  auth_user_id uuid NOT NULL UNIQUE,
  name character varying NOT NULL,
  org_id bigint NOT NULL,
  specialty text DEFAULT 'General'::text,
  license_number character varying,
  is_active boolean DEFAULT false,
  status character varying DEFAULT 'pending'::character varying,
  last_login timestamp with time zone,
  approved_by bigint,
  approved_at timestamp with time zone,
  email character varying,
  phone character varying,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT doctor_pkey PRIMARY KEY (id),
  CONSTRAINT doctor_org_id_fkey FOREIGN KEY (org_id) REFERENCES public.org(id),
  CONSTRAINT doctor_auth_user_id_fkey FOREIGN KEY (auth_user_id) REFERENCES auth.users(id),
  CONSTRAINT doctor_approved_by_fkey FOREIGN KEY (approved_by) REFERENCES public.admin(id)
);
CREATE TABLE public.org (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  name character varying NOT NULL UNIQUE,
  address text,
  phone text,
  email text,
  secret_code text,
  CONSTRAINT org_pkey PRIMARY KEY (id)
);
CREATE TABLE public.patients (
  id integer GENERATED ALWAYS AS IDENTITY NOT NULL,
  patient_id character varying UNIQUE,
  name character varying NOT NULL,
  age smallint,
  gender character varying,
  date_of_birth date NOT NULL,
  date_registered date DEFAULT CURRENT_DATE,
  medical_record_number character varying,
  phone numeric,
  address text,
  emergency_contact numeric,
  risk_factors jsonb,
  last_test_date timestamp with time zone,
  created_by bigint,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT patients_pkey PRIMARY KEY (id),
  CONSTRAINT patients_created_by_fkey FOREIGN KEY (created_by) REFERENCES public.doctor(id)
);
CREATE TABLE public.prediction_details (
  id integer GENERATED ALWAYS AS IDENTITY NOT NULL,
  prediction_id integer NOT NULL,
  species_detected character varying,
  parasite_count integer,
  grad_cam_path character varying,
  parasite_stage text,
  attention_regions jsonb,
  image_quality_score integer,
  analysis_duration_sec bigint,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT prediction_details_pkey PRIMARY KEY (id),
  CONSTRAINT prediction_details_prediction_id_fkey FOREIGN KEY (prediction_id) REFERENCES public.predictions(id)
);
CREATE TABLE public.prediction_history (
  id bigint GENERATED ALWAYS AS IDENTITY NOT NULL,
  sample_id integer NOT NULL,
  doctor_id bigint,
  endpoint_used character varying,
  request_payload jsonb,
  status character varying NOT NULL,
  response_payload jsonb,
  error_message text,
  processing_time_ms integer,
  model_version smallint,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT prediction_history_pkey PRIMARY KEY (id),
  CONSTRAINT prediction_history_sample_id_fkey FOREIGN KEY (sample_id) REFERENCES public.blood_samples(id),
  CONSTRAINT prediction_history_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctor(id)
);
CREATE TABLE public.predictions (
  id integer GENERATED ALWAYS AS IDENTITY NOT NULL,
  sample_id integer NOT NULL,
  predicted_class text NOT NULL,
  confidence_score numeric CHECK (confidence_score >= 0::numeric AND confidence_score <= 1::numeric),
  prediction_date date DEFAULT CURRENT_DATE,
  model_version smallint,
  doctor_id bigint,
  probabilities jsonb,
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT predictions_pkey PRIMARY KEY (id),
  CONSTRAINT predictions_sample_id_fkey FOREIGN KEY (sample_id) REFERENCES public.blood_samples(id),
  CONSTRAINT predictions_doctor_id_fkey FOREIGN KEY (doctor_id) REFERENCES public.doctor(id)
);