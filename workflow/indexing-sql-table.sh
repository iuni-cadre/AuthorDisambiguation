sqldb=$1

sqlite3 $sqldb 'create index citing_index on data(citing);'
sqlite3 $sqldb 'create index cited_index on data(cited);'
sqlite3 $sqldb 'ALTER TABLE data RENAME TO citation_table;'
