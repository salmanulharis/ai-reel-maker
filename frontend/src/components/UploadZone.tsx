import { useRef, useState } from "react";
import { fill, messages } from "../i18n";
import { formatBytes } from "../services/api";
import { UploadIcon } from "./Feedback";

const ACCEPT = ".mp4,.mov,.mkv,.avi,.webm,.m4v,video/*";
const MAX_FILES = 8;

function asVideoFiles(list: FileList | File[] | null): File[] {
  if (!list) return [];
  return Array.from(list).filter((file) => file.type.startsWith("video/") || /\.(mp4|mov|mkv|avi|webm|m4v)$/i.test(file.name));
}

export function UploadZone({
  onFiles,
  files,
  onRemove,
}: {
  onFiles: (files: File[]) => void;
  files: File[];
  onRemove: (index: number) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);

  function add(incoming: File[]) {
    if (!incoming.length) return;
    onFiles(incoming.slice(0, MAX_FILES));
  }

  return (
    <div>
      <div
        className={`upload ${drag ? "drag" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setDrag(false);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          add(asVideoFiles(e.dataTransfer.files));
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            input.current?.click();
          }
        }}
        aria-label={`${messages.upload.drop}. ${messages.upload.browse}`}
        aria-describedby="upload-formats"
      >
        <div>
          <div className="upload-icon">
            <UploadIcon size={20} />
          </div>
          <h3>{messages.upload.drop}</h3>
          <p>
            {messages.upload.or} <span className="browse">{messages.upload.browse}</span>
          </p>
          <p className="formats" id="upload-formats">
            {messages.upload.formats}
          </p>
        </div>
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          multiple
          onChange={(e) => {
            add(asVideoFiles(e.target.files));
            e.target.value = "";
          }}
        />
      </div>
      {files.length > 0 && (
        <div className="file-list">
          <p className="file-list-title">
            {messages.upload.files} ({files.length}/{MAX_FILES})
          </p>
          <ol>
            {files.map((file, index) => (
              <li key={`${file.name}-${file.size}-${index}`}>
                <span className="file-name" title={file.name}>
                  {fill(messages.upload.video, { n: index + 1 })}: {file.name}
                  <span className="file-size">{formatBytes(file.size)}</span>
                </span>
                <button
                  className="btn ghost sm"
                  type="button"
                  aria-label={`${messages.upload.removeFile} ${file.name}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onRemove(index);
                  }}
                >
                  {messages.upload.removeFile}
                </button>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
