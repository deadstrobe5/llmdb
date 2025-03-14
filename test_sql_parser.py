#!/usr/bin/env python3
import os
import sys
import argparse
import json
from typing import List, Dict, Any, Optional

# Import the SQLFileParser
from src.db.sql_file_parser import SQLFileParser

def display_table_schema(parser: SQLFileParser, table_name: str) -> None:
    """Display the schema for a specific table"""
    schema = parser.get_table_schema(table_name)
    
    print(f"\n{'='*80}")
    print(f"TABLE: {table_name}")
    print(f"{'='*80}")
    
    # Display columns
    print("\nCOLUMNS:")
    print(f"{'Name':<20} {'Type':<20} {'Nullable':<10} {'Default':<15} {'PK':<5}")
    print(f"{'-'*20} {'-'*20} {'-'*10} {'-'*15} {'-'*5}")
    
    for col in schema.get('columns', []):
        print(f"{col['name']:<20} {col['type']:<20} {str(col['nullable']):<10} {str(col['default']):<15} {str(col['primary_key']):<5}")
    
    # Display primary key
    pk = schema.get('primary_key', {})
    if pk and pk.get('constrained_columns'):
        print("\nPRIMARY KEY:")
        print(", ".join(pk.get('constrained_columns', [])))
    
    # Display foreign keys
    fkeys = schema.get('foreign_keys', [])
    if fkeys:
        print("\nFOREIGN KEYS:")
        for fk in fkeys:
            cols = ", ".join(fk.get('constrained_columns', []))
            ref_table = fk.get('referred_table', '')
            ref_cols = ", ".join(fk.get('referred_columns', []))
            print(f"{cols} -> {ref_table}({ref_cols})")

def display_sample_data(parser: SQLFileParser, table_name: str, limit: int = 5) -> None:
    """Display sample data for a specific table"""
    sample_data = parser.extract_sample_data(table_name, limit)
    
    if not sample_data:
        print("\nNo sample data found for this table.")
        return
    
    print(f"\nSAMPLE DATA ({len(sample_data)} rows):")
    
    # Get all keys from the samples to construct headers
    # Filter out keys with no values
    headers = set()
    for row in sample_data:
        # Only include keys that have non-empty values in at least one row
        for key, value in row.items():
            if value is not None and (not isinstance(value, str) or value.strip()):
                headers.add(key)
    
    headers = sorted(list(headers))
    
    # Calculate dynamic column width
    col_width = 20
    
    # Display headers
    header_row = " | ".join(f"{h[:col_width-3]+'...' if len(h) > col_width else h:<{col_width}}" for h in headers)
    print("\n" + header_row)
    print("-" * len(header_row))
    
    # Display data rows
    for row in sample_data:
        values = []
        for key in headers:
            value = row.get(key, "")
            
            # Skip None or empty values
            if value is None:
                values.append(f"{'NULL':<{col_width}}")
                continue
                
            # Format different types of values
            if isinstance(value, (int, float)):
                # Format numbers
                values.append(f"{str(value):<{col_width}}")
            elif isinstance(value, str):
                # Truncate and clean string values
                clean_value = value.strip()
                if len(clean_value) > col_width - 3 and col_width > 3:
                    clean_value = clean_value[:col_width-3] + "..."
                values.append(f"{clean_value:<{col_width}}")
            else:
                # Other types
                values.append(f"{str(value)[:col_width-3]+'...' if len(str(value)) > col_width else str(value):<{col_width}}")
        
        print(" | ".join(values))

def list_tables(parser: SQLFileParser) -> None:
    """List all tables in the SQL file"""
    tables = parser.get_all_table_names()
    
    print(f"\nFound {len(tables)} tables:")
    print(f"{'='*80}")
    
    # Display in multiple columns
    cols = 3
    col_width = 25
    
    for i in range(0, len(tables), cols):
        row = tables[i:i+cols]
        print("".join(f"{table:<{col_width}}" for table in row))

def export_json_sample(parser: SQLFileParser, table_name: str, limit: int = 5, output_path: Optional[str] = None) -> None:
    """Export sample data as JSON for vector database processing"""
    sample_data = parser.extract_sample_data(table_name, limit)
    
    if not sample_data:
        print(f"No sample data found for table: {table_name}")
        return
    
    # Clean the data further for vector database use
    clean_samples = []
    for row in sample_data:
        # Remove any None or empty string values
        clean_row = {
            k: v for k, v in row.items() 
            if v is not None and (not isinstance(v, str) or v.strip())
        }
        
        if clean_row:
            clean_samples.append(clean_row)
    
    # Generate output filename if not provided
    if not output_path:
        output_path = f"{table_name}_sample.json"
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\nExported {len(clean_samples)} clean samples to {output_path}")

def export_vector_chunks(parser: SQLFileParser, table_name: str, limit: int = 5, output_path: Optional[str] = None) -> None:
    """
    Export LLM-friendly vector chunks for the table.
    These chunks are optimized for vector database storage and LLM understanding.
    """
    vector_chunks = parser.create_vector_chunks(table_name, limit)
    
    if not vector_chunks:
        print(f"No data found for creating vector chunks from table: {table_name}")
        return
    
    # Generate output filename if not provided
    if not output_path:
        output_path = f"{table_name}_vector_chunks.json"
    
    # Write to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vector_chunks, f, ensure_ascii=False, indent=2)
    
    # Display a sample of the first chunk's content
    if vector_chunks:
        print(f"\nExported {len(vector_chunks)} vector chunks to {output_path}")
        print("\nSample vector chunk content:")
        print("-" * 80)
        print(vector_chunks[0]["content"])
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description='Test SQL file parser')
    parser.add_argument('sql_file', help='Path to SQL file to parse')
    parser.add_argument('--table', '-t', help='Specific table to analyze')
    parser.add_argument('--list-tables', '-l', action='store_true', help='List all tables')
    parser.add_argument('--sample-limit', '-s', type=int, default=5, help='Number of sample rows to display')
    parser.add_argument('--export-json', '-e', help='Export sample data as JSON for vector DB (provide output path or leave empty for default)')
    parser.add_argument('--vector-chunks', '-v', action='store_true', help='Export optimized vector chunks for LLM')
    
    args = parser.parse_args()
    
    # Check if SQL file exists
    if not os.path.exists(args.sql_file):
        print(f"Error: SQL file '{args.sql_file}' not found")
        return 1
    
    # Initialize parser
    sql_parser = SQLFileParser(args.sql_file)
    
    # If table is specified, analyze it
    if args.table:
        display_table_schema(sql_parser, args.table)
        
        # Export vector chunks if requested
        if args.vector_chunks:
            export_vector_chunks(sql_parser, args.table, args.sample_limit)
        # Export JSON if requested
        elif args.export_json is not None:
            export_json_sample(sql_parser, args.table, args.sample_limit, 
                              args.export_json if args.export_json else None)
        # Otherwise display sample data
        else:
            display_sample_data(sql_parser, args.table, args.sample_limit)
    # Otherwise list all tables
    elif args.list_tables or not args.table:
        list_tables(sql_parser)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 