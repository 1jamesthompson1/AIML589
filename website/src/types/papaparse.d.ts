declare module 'papaparse' {
  export interface ParseResult<T = any> {
    data: T[];
    errors: unknown[];
    meta: { delimiter: string; linebreak: string; aborted: boolean; truncated: boolean; cursor: number };
  }

  export interface ParseConfig<T = any> {
    header?: boolean;
    dynamicTyping?: boolean | ((field: string) => boolean);
    skipEmptyLines?: boolean | 'greedy';
    transform?: (value: string, field: string | number) => string;
    transformHeader?: (header: string, index: number) => string;
    delimiter?: string;
    newline?: string;
    quoteChar?: string;
    comments?: string | boolean;
    download?: boolean;
    worker?: boolean;
    preview?: number;
    step?: (results: ParseResult<T>, parser?: unknown) => void;
    complete?: (results: ParseResult<T>, file?: unknown) => void;
    error?: (error: unknown) => void;
  }

  export function parse<T = any>(input: string | File | object, config?: ParseConfig<T>): ParseResult<T>;
  export function unparse(data: unknown, config?: unknown): string;

  const Papa: { parse: typeof parse; unparse: typeof unparse };
  export default Papa;
}
