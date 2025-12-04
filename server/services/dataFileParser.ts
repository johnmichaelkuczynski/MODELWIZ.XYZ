import Papa from 'papaparse';
import ExcelJS from 'exceljs';
import Database from 'better-sqlite3';
import * as fs from 'fs';
import * as path from 'path';

// Parquet is optional - may not have types
let parquet: any;
try {
  parquet = require('parquetjs');
} catch (e) {
  console.warn('Parquet support not available');
}

export interface DataFileResult {
  success: boolean;
  filename: string;
  fileType: string;
  rowCount: number;
  columnCount: number;
  columns: string[];
  preview: Record<string, any>[]; // First 10 rows
  schema: { name: string; type: string }[];
  rawData?: Record<string, any>[]; // Full data if small enough
  error?: string;
}

export interface JupyterNotebookResult {
  success: boolean;
  filename: string;
  fileType: 'jupyter';
  cellCount: number;
  codeCells: string[];
  markdownCells: string[];
  outputs: string[];
  rowCount: number;
  columnCount: number;
  columns: string[];
  schema: { name: string; type: string }[];
  preview: Record<string, any>[];
  error?: string;
}

function inferDataType(values: any[]): string {
  const nonNull = values.filter(v => v !== null && v !== undefined && v !== '');
  if (nonNull.length === 0) return 'unknown';
  
  const sample = nonNull.slice(0, 100);
  
  let allNumbers = true;
  let allIntegers = true;
  let allBooleans = true;
  let allDates = true;
  
  for (const val of sample) {
    const strVal = String(val).trim();
    
    // Check boolean
    if (!['true', 'false', '1', '0', 'yes', 'no'].includes(strVal.toLowerCase())) {
      allBooleans = false;
    }
    
    // Check number
    const numVal = Number(strVal);
    if (isNaN(numVal)) {
      allNumbers = false;
      allIntegers = false;
    } else if (!Number.isInteger(numVal)) {
      allIntegers = false;
    }
    
    // Check date
    const dateVal = new Date(strVal);
    if (isNaN(dateVal.getTime()) || strVal.length < 6) {
      allDates = false;
    }
  }
  
  if (allBooleans) return 'boolean';
  if (allIntegers) return 'integer';
  if (allNumbers) return 'float';
  if (allDates) return 'datetime';
  return 'string';
}

export async function parseCSV(buffer: Buffer, filename: string): Promise<DataFileResult> {
  try {
    const content = buffer.toString('utf-8');
    
    const result = Papa.parse(content, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: true,
    });
    
    if (result.errors.length > 0 && result.data.length === 0) {
      throw new Error(result.errors.map(e => e.message).join('; '));
    }
    
    const data = result.data as Record<string, any>[];
    const columns = result.meta.fields || [];
    
    // Infer schema
    const schema = columns.map(col => ({
      name: col,
      type: inferDataType(data.map(row => row[col])),
    }));
    
    return {
      success: true,
      filename,
      fileType: 'csv',
      rowCount: data.length,
      columnCount: columns.length,
      columns,
      preview: data.slice(0, 10),
      schema,
      rawData: data.length <= 1000 ? data : undefined,
    };
  } catch (error: any) {
    return {
      success: false,
      filename,
      fileType: 'csv',
      rowCount: 0,
      columnCount: 0,
      columns: [],
      preview: [],
      schema: [],
      error: error.message,
    };
  }
}

export async function parseExcel(buffer: Buffer, filename: string): Promise<DataFileResult> {
  try {
    const workbook = new ExcelJS.Workbook();
    await workbook.xlsx.load(buffer);
    
    const worksheet = workbook.worksheets[0];
    if (!worksheet) {
      throw new Error('No worksheets found in Excel file');
    }
    
    const data: Record<string, any>[] = [];
    let columns: string[] = [];
    
    worksheet.eachRow((row, rowNumber) => {
      if (rowNumber === 1) {
        // Header row - extract values as array
        const values = row.values as any[];
        if (values) {
          columns = values.slice(1).map((v: any, i: number) => String(v || `Column${i + 1}`));
        }
      } else {
        const rowData: Record<string, any> = {};
        const values = row.values as any[];
        if (values) {
          values.slice(1).forEach((value: any, index: number) => {
            if (index < columns.length) {
              rowData[columns[index]] = value;
            }
          });
        }
        data.push(rowData);
      }
    });
    
    const schema = columns.map(col => ({
      name: col,
      type: inferDataType(data.map(row => row[col])),
    }));
    
    return {
      success: true,
      filename,
      fileType: 'excel',
      rowCount: data.length,
      columnCount: columns.length,
      columns,
      preview: data.slice(0, 10),
      schema,
      rawData: data.length <= 1000 ? data : undefined,
    };
  } catch (error: any) {
    return {
      success: false,
      filename,
      fileType: 'excel',
      rowCount: 0,
      columnCount: 0,
      columns: [],
      preview: [],
      schema: [],
      error: error.message,
    };
  }
}

export async function parseJSON(buffer: Buffer, filename: string): Promise<DataFileResult> {
  try {
    const content = buffer.toString('utf-8');
    let parsed = JSON.parse(content);
    
    // Handle different JSON structures
    let data: Record<string, any>[];
    
    if (Array.isArray(parsed)) {
      data = parsed;
    } else if (typeof parsed === 'object' && parsed !== null) {
      // Look for array properties
      const arrayKeys = Object.keys(parsed).filter(k => Array.isArray(parsed[k]));
      if (arrayKeys.length > 0) {
        // Use the first array found (common pattern: { data: [...] })
        data = parsed[arrayKeys[0]];
      } else {
        // Wrap single object in array
        data = [parsed];
      }
    } else {
      throw new Error('JSON must be an array or object with array data');
    }
    
    if (data.length === 0) {
      return {
        success: true,
        filename,
        fileType: 'json',
        rowCount: 0,
        columnCount: 0,
        columns: [],
        preview: [],
        schema: [],
      };
    }
    
    // Get all unique keys from all objects
    const columnSet = new Set<string>();
    data.forEach(obj => {
      Object.keys(obj).forEach(key => columnSet.add(key));
    });
    const columns = Array.from(columnSet);
    
    const schema = columns.map(col => ({
      name: col,
      type: inferDataType(data.map(row => row[col])),
    }));
    
    return {
      success: true,
      filename,
      fileType: 'json',
      rowCount: data.length,
      columnCount: columns.length,
      columns,
      preview: data.slice(0, 10),
      schema,
      rawData: data.length <= 1000 ? data : undefined,
    };
  } catch (error: any) {
    return {
      success: false,
      filename,
      fileType: 'json',
      rowCount: 0,
      columnCount: 0,
      columns: [],
      preview: [],
      schema: [],
      error: error.message,
    };
  }
}

const MAX_ROWS_READ = 1000;
const PREVIEW_ROWS = 10;

export async function parseParquet(buffer: Buffer, filename: string): Promise<DataFileResult> {
  if (!parquet) {
    return {
      success: false,
      filename,
      fileType: 'parquet',
      rowCount: 0,
      columnCount: 0,
      columns: [],
      preview: [],
      schema: [],
      error: 'Parquet format is not supported in this environment. Please convert to CSV or JSON.',
    };
  }
  
  const tempPath = path.join('/tmp', `parquet_${Date.now()}.parquet`);
  
  try {
    fs.writeFileSync(tempPath, buffer);
    
    const reader = await parquet.ParquetReader.openFile(tempPath);
    const cursor = reader.getCursor();
    
    const data: Record<string, any>[] = [];
    let record = null;
    
    while (record = await cursor.next()) {
      data.push(record);
      if (data.length >= MAX_ROWS_READ) break;
    }
    
    await reader.close();
    
    const columns = data.length > 0 ? Object.keys(data[0]) : [];
    
    const schema = columns.map(col => ({
      name: col,
      type: inferDataType(data.map(row => row[col])),
    }));
    
    return {
      success: true,
      filename,
      fileType: 'parquet',
      rowCount: data.length,
      columnCount: columns.length,
      columns,
      preview: data.slice(0, PREVIEW_ROWS),
      schema,
      rawData: data.length <= MAX_ROWS_READ ? data : undefined,
    };
  } catch (error: any) {
    return {
      success: false,
      filename,
      fileType: 'parquet',
      rowCount: 0,
      columnCount: 0,
      columns: [],
      preview: [],
      schema: [],
      error: error.message,
    };
  } finally {
    try {
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
      }
    } catch (e) {
      console.warn('Failed to clean up temp parquet file:', e);
    }
  }
}

export async function parseSQLite(buffer: Buffer, filename: string): Promise<DataFileResult & { tables?: string[] }> {
  const tempPath = path.join('/tmp', `sqlite_${Date.now()}.db`);
  let db: any = null;
  
  try {
    fs.writeFileSync(tempPath, buffer);
    
    db = new Database(tempPath, { readonly: true });
    
    const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").all() as { name: string }[];
    const tableNames = tables.map(t => t.name);
    
    if (tableNames.length === 0) {
      throw new Error('No tables found in SQLite database');
    }
    
    const mainTable = tableNames[0];
    const data = db.prepare(`SELECT * FROM "${mainTable}" LIMIT ${MAX_ROWS_READ}`).all() as Record<string, any>[];
    const countResult = db.prepare(`SELECT COUNT(*) as count FROM "${mainTable}"`).get() as { count: number };
    const totalRows = countResult.count;
    
    const columns = data.length > 0 ? Object.keys(data[0]) : [];
    
    const pragmaInfo = db.prepare(`PRAGMA table_info("${mainTable}")`).all() as { name: string; type: string }[];
    const schema = pragmaInfo.map(col => ({
      name: col.name,
      type: col.type.toLowerCase() || 'text',
    }));
    
    return {
      success: true,
      filename,
      fileType: 'sqlite',
      rowCount: totalRows,
      columnCount: columns.length,
      columns,
      preview: data.slice(0, PREVIEW_ROWS),
      schema,
      rawData: data.length <= MAX_ROWS_READ ? data : undefined,
      tables: tableNames,
    };
  } catch (error: any) {
    return {
      success: false,
      filename,
      fileType: 'sqlite',
      rowCount: 0,
      columnCount: 0,
      columns: [],
      preview: [],
      schema: [],
      error: error.message,
    };
  } finally {
    try {
      if (db) db.close();
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
      }
    } catch (e) {
      console.warn('Failed to clean up temp sqlite file:', e);
    }
  }
}

export async function parseJupyterNotebook(buffer: Buffer, filename: string): Promise<JupyterNotebookResult> {
  try {
    const content = buffer.toString('utf-8');
    const notebook = JSON.parse(content);
    
    if (!notebook.cells || !Array.isArray(notebook.cells)) {
      throw new Error('Invalid Jupyter notebook format');
    }
    
    const codeCells: string[] = [];
    const markdownCells: string[] = [];
    const outputs: string[] = [];
    
    for (const cell of notebook.cells) {
      const source = Array.isArray(cell.source) ? cell.source.join('') : cell.source;
      
      if (cell.cell_type === 'code') {
        codeCells.push(source);
        
        if (cell.outputs && Array.isArray(cell.outputs)) {
          for (const output of cell.outputs) {
            if (output.text) {
              outputs.push(Array.isArray(output.text) ? output.text.join('') : output.text);
            } else if (output.data && output.data['text/plain']) {
              const text = output.data['text/plain'];
              outputs.push(Array.isArray(text) ? text.join('') : text);
            }
          }
        }
      } else if (cell.cell_type === 'markdown') {
        markdownCells.push(source);
      }
    }
    
    return {
      success: true,
      filename,
      fileType: 'jupyter',
      cellCount: notebook.cells.length,
      codeCells,
      markdownCells,
      outputs: outputs.slice(0, 20),
      rowCount: codeCells.length,
      columnCount: 0,
      columns: ['code', 'type'],
      schema: [
        { name: 'code', type: 'string' },
        { name: 'type', type: 'string' }
      ],
      preview: codeCells.slice(0, PREVIEW_ROWS).map((code, i) => ({ code, type: 'code', index: i })),
    };
  } catch (error: any) {
    return {
      success: false,
      filename,
      fileType: 'jupyter',
      cellCount: 0,
      codeCells: [],
      markdownCells: [],
      outputs: [],
      rowCount: 0,
      columnCount: 0,
      columns: [],
      schema: [],
      preview: [],
      error: error.message,
    };
  }
}

export async function parseDataFile(
  buffer: Buffer, 
  filename: string, 
  mimeType?: string
): Promise<DataFileResult | JupyterNotebookResult> {
  const ext = path.extname(filename).toLowerCase();
  
  switch (ext) {
    case '.csv':
      return parseCSV(buffer, filename);
    case '.xlsx':
    case '.xls':
      return parseExcel(buffer, filename);
    case '.json':
      return parseJSON(buffer, filename);
    case '.parquet':
      return parseParquet(buffer, filename);
    case '.db':
    case '.sqlite':
    case '.sqlite3':
      return parseSQLite(buffer, filename);
    case '.ipynb':
      return parseJupyterNotebook(buffer, filename);
    default:
      // Try to infer from content
      const content = buffer.toString('utf-8', 0, 1000);
      if (content.trim().startsWith('{') || content.trim().startsWith('[')) {
        return parseJSON(buffer, filename);
      }
      if (content.includes(',') && content.includes('\n')) {
        return parseCSV(buffer, filename);
      }
      return {
        success: false,
        filename,
        fileType: 'unknown',
        rowCount: 0,
        columnCount: 0,
        columns: [],
        preview: [],
        schema: [],
        error: `Unsupported file format: ${ext}`,
      };
  }
}
