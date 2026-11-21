-- Privacy policy and terms of use, every version ever written.
--
-- Lives in the platform Supabase project (the one that holds the logins), not
-- in each workspace's database: there is one policy for the whole service.
-- Apply once:  psql "$PLATFORM_DATABASE_URL" -f app/admin/legal_schema.sql
-- (safe to re-run).
--
-- Who touches it:
--   * the admin site (admin.continuity.marketing/legal) reads and writes it
--     through Supabase's REST API with the service key;
--   * the public pages (continuity.marketing/privacy, /terms) read published
--     rows through the REST API with the publishable key (the anon role),
--     which the policy below limits to published versions.
--
-- A version starts as a draft, which can be edited or deleted. Publishing
-- fixes it: after that the trigger refuses any change, so the history is a
-- faithful record of what was in force when.

create table if not exists public.legal_documents (
    id             bigint generated always as identity primary key,
    kind           text        not null check (kind in ('privacy', 'terms')),
    version        integer     not null check (version > 0),
    status         text        not null default 'draft' check (status in ('draft', 'published')),
    title          text        not null,
    summary        text        not null default '',   -- what changed, in a sentence or two
    body_md        text        not null default '',   -- what the admin edits
    body_html      text        not null default '',   -- rendered from body_md on every save
    effective_date date,                               -- set when published; may be in the future
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now(),
    published_at   timestamptz,
    published_by   text,
    unique (kind, version),
    check (status = 'draft' or (effective_date is not null and published_at is not null))
);

create index if not exists legal_documents_current
    on public.legal_documents (kind, effective_date desc, version desc)
    where status = 'published';

-- At most one draft per document.
create unique index if not exists legal_documents_one_draft
    on public.legal_documents (kind) where status = 'draft';

create or replace function public.legal_documents_guard() returns trigger
language plpgsql as $$
begin
    if old.status = 'published' then
        raise exception 'legal_documents: version % of % is published and cannot be changed',
            old.version, old.kind;
    end if;
    if tg_op = 'DELETE' then
        return old;
    end if;
    new.updated_at := now();
    return new;
end $$;

drop trigger if exists legal_documents_guard on public.legal_documents;
create trigger legal_documents_guard
    before update or delete on public.legal_documents
    for each row execute function public.legal_documents_guard();

alter table public.legal_documents enable row level security;

drop policy if exists "published legal documents are public" on public.legal_documents;
create policy "published legal documents are public"
    on public.legal_documents for select
    to anon, authenticated
    using (status = 'published' and effective_date <= current_date);

grant select on public.legal_documents to anon, authenticated;
