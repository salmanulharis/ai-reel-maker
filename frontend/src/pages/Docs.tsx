import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { messages } from "../i18n";
import {
  DOC_ARTICLES,
  DOC_CATEGORIES,
  FIRST_ARTICLE,
  searchDocs,
  type DocArticleId,
  type DocBlock,
} from "../docs/content";

const copy = messages.docs;

function isArticle(value: string | null): value is DocArticleId {
  return Boolean(value && value in DOC_ARTICLES);
}

/** **bold** and `code` inside help text. */
export function InlineText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return (
    <>
      {parts.map((part, index) =>
        part.startsWith("**") ? (
          <strong key={index}>{part.slice(2, -2)}</strong>
        ) : part.startsWith("`") ? (
          <code key={index}>{part.slice(1, -1)}</code>
        ) : (
          <span key={index}>{part}</span>
        )
      )}
    </>
  );
}

function Block({ block }: { block: DocBlock }) {
  switch (block.kind) {
    case "p":
      return (
        <p>
          <InlineText text={block.text} />
        </p>
      );
    case "h":
      return <h4 className="doc-subheading">{block.text}</h4>;
    case "steps":
    case "list": {
      const Tag = block.kind === "steps" ? "ol" : "ul";
      return (
        <Tag className={block.kind === "steps" ? "doc-steps" : "doc-list"}>
          {block.items.map((item) => (
            <li key={item}>
              <InlineText text={item} />
            </li>
          ))}
        </Tag>
      );
    }
    case "tip":
    case "note":
    case "warning":
      return (
        <div className={`doc-callout ${block.kind}`} role="note">
          <strong>{copy.callouts[block.kind]}</strong> <InlineText text={block.text} />
        </div>
      );
    case "code":
      return (
        <figure className="doc-code">
          {block.label && <figcaption>{block.label}</figcaption>}
          <pre>
            <code>{block.text}</code>
          </pre>
        </figure>
      );
    case "table":
      return (
        <div className="table-wrap doc-table">
          <table className="table">
            <thead>
              <tr>
                {block.head.map((cell) => (
                  <th key={cell} scope="col">
                    {cell}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row) => (
                <tr key={row.join("|")}>
                  {row.map((cell, index) => (
                    <td key={index}>
                      <InlineText text={cell} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "settings":
      return (
        <p className="doc-action">
          <Link className="btn sm" to={`/settings?section=${block.section}`}>
            {block.label} →
          </Link>
        </p>
      );
    case "see":
      return isArticle(block.article) ? (
        <p className="doc-action">
          <Link to={`/help?article=${block.article}`}>{block.label} →</Link>
        </p>
      ) : null;
  }
}

/** "Learn more" link from a setting or feature to its help article. */
export function DocLink({ article, children }: { article: DocArticleId; children?: ReactNode }) {
  return (
    <Link className="doc-link" to={`/help?article=${article}`}>
      {children ?? copy.learnMore}
    </Link>
  );
}

export default function Docs() {
  const [params, setParams] = useSearchParams();
  const current: DocArticleId = isArticle(params.get("article")) ? (params.get("article") as DocArticleId) : FIRST_ARTICLE;
  const [query, setQuery] = useState("");
  const searchId = useId();
  const headingRef = useRef<HTMLHeadingElement>(null);
  const results = query.trim() ? searchDocs(query) : null;
  const article = DOC_ARTICLES[current];

  function open(id: DocArticleId) {
    setQuery("");
    setParams(id === FIRST_ARTICLE ? {} : { article: id });
  }

  // Move focus to the heading when another article opens (not on first load), for keyboard and
  // screen-reader users.
  const firstRender = useRef(true);
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    headingRef.current?.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }, [current]);

  return (
    <div className="settings-layout docs-layout">
      <nav className="settings-nav" aria-label={copy.navLabel}>
        {DOC_CATEGORIES.map((category) => (
          <div key={category.title} className="settings-nav-group">
            <p className="settings-nav-heading" aria-hidden="true">
              {category.title}
            </p>
            <ul>
              {category.articles.map((id) => (
                <li key={id}>
                  <button
                    type="button"
                    className={`settings-nav-item ${!results && id === current ? "active" : ""}`}
                    aria-current={!results && id === current ? "page" : undefined}
                    onClick={() => open(id)}
                  >
                    {DOC_ARTICLES[id].title}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      <div className="settings-body">
        <div className="field docs-search" role="search">
          <label htmlFor={searchId} className="sr-only">
            {copy.search}
          </label>
          <input
            id={searchId}
            type="search"
            placeholder={copy.searchPlaceholder}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        {results ? (
          <section className="panel" aria-live="polite">
            <h3 className="sub-title settings-title">
              {results.length ? copy.results.replace("{n}", String(results.length)) : copy.noResults}
            </h3>
            {results.length ? (
              <ul className="doc-results">
                {results.map((id) => (
                  <li key={id}>
                    <button type="button" className="link-button" onClick={() => open(id)}>
                      {DOC_ARTICLES[id].title}
                    </button>
                    <p className="hint">{DOC_ARTICLES[id].summary}</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="hint">{copy.noResultsHint}</p>
            )}
          </section>
        ) : (
          <article className="panel doc-article" aria-labelledby="doc-title" key={current}>
            <h3 id="doc-title" className="sub-title settings-title" tabIndex={-1} ref={headingRef}>
              {article.title}
            </h3>
            <p className="panel-intro">{article.summary}</p>
            {article.blocks.map((block, index) => (
              <Block key={index} block={block} />
            ))}
          </article>
        )}
      </div>
    </div>
  );
}
